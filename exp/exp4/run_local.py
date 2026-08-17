#!/usr/bin/env python3
"""Run local Qwen on Aider Polyglot with execution-grounded repair feedback."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluations.local_sglang import LocalSGLangGenerator
from exp.exp4.polyglot import (
    POLYGLOT_COMMIT,
    WHOLE_FILE_SYSTEM_PROMPT,
    PolyglotTask,
    discover_tasks,
    parse_whole_file_edit,
    test_feedback,
    verify_edit,
    write_manifest,
)


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _command_output(command: Sequence[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def _provenance(
    *,
    args: argparse.Namespace,
    source_root: Path,
    generator: LocalSGLangGenerator | None,
) -> dict[str, Any]:
    payload = {
        "flashtrace_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "flashtrace_dirty": bool(
            _command_output(["git", "status", "--short"])
        ),
        "polyglot_commit": _command_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"]
        ),
        "aider_harness_commit": _command_output(
            ["git", "-C", "data/external/aider", "rev-parse", "HEAD"]
        ),
        "docker_image": args.image,
        "docker_image_id": _command_output(
            ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"]
        ),
        "runner_file_sha256": {
            str(path): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("polyglot.py"),
                Path(__file__).with_name("Dockerfile"),
            )
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                    "-i",
                    args.device.split(":")[-1],
                ]
            )
            if generator is not None
            else (
                torch.cuda.get_device_name(int(args.device.split(":")[-1]))
                if torch.cuda.is_available()
                else None
            )
        ),
    }
    if generator is not None:
        payload["sglang_server"] = generator.provenance()
    return payload


def _strip_eos(text: str, eos_token: str | None) -> str:
    if eos_token and text.endswith(eos_token):
        return text[: -len(eos_token)]
    return text


def _completion_spans(
    completion: str,
) -> dict[str, list[int] | str | None]:
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")
    if think_start >= 0 and think_end >= think_start:
        reasoning = [think_start + len("<think>"), think_end]
        action_start = think_end + len("</think>")
        while action_start < len(completion) and completion[action_start].isspace():
            action_start += 1
        return {
            "reasoning_char_span_half_open": reasoning,
            "action_char_span_half_open": [action_start, len(completion)],
            "reasoning_boundary": "explicit_open_and_close",
        }
    if think_start < 0 and think_end >= 0:
        action_start = think_end + len("</think>")
        while action_start < len(completion) and completion[action_start].isspace():
            action_start += 1
        return {
            "reasoning_char_span_half_open": [0, think_end],
            "action_char_span_half_open": [action_start, len(completion)],
            "reasoning_boundary": "implicit_open_explicit_close",
        }
    return {
        "reasoning_char_span_half_open": None,
        "action_char_span_half_open": None,
        "reasoning_boundary": "missing_close",
    }


@torch.no_grad()
def generate_turn(
    model: Any,
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    max_new_tokens: int,
    generator: LocalSGLangGenerator | None = None,
) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=True,
    )
    if generator is not None:
        turn = generator.generate(rendered, max_new_tokens)
        return {
            **turn,
            **_completion_spans(turn["completion"]),
        }
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    seconds = time.perf_counter() - started
    new_ids = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    raw = tokenizer.decode(
        new_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    completion = _strip_eos(raw, tokenizer.eos_token)
    return {
        "rendered_prompt": rendered,
        "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
        "completion_token_ids_with_eos_if_emitted": new_ids,
        "raw_completion_with_eos": raw,
        "completion": completion,
        "stopped_on_eos": bool(
            new_ids
            and tokenizer.eos_token_id is not None
            and new_ids[-1] == tokenizer.eos_token_id
        ),
        "finish_reason": (
            "eos"
            if new_ids
            and tokenizer.eos_token_id is not None
            and new_ids[-1] == tokenizer.eos_token_id
            else "max_new_tokens"
        ),
        "generation_seconds": seconds,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        ),
        **_completion_spans(completion),
    }


def _stratified_selection(
    tasks: Sequence[PolyglotTask],
    count: int | None,
    *,
    seed: int,
) -> list[PolyglotTask]:
    if count is None or count >= len(tasks):
        return list(tasks)
    if count < 1:
        raise ValueError("count must be positive")

    by_language: dict[str, list[PolyglotTask]] = {}
    for task in tasks:
        by_language.setdefault(task.language, []).append(task)
    languages = sorted(by_language)
    base, remainder = divmod(count, len(languages))
    selected = []
    for language_index, language in enumerate(languages):
        take = base + int(language_index < remainder)
        candidates = sorted(by_language[language], key=lambda task: len(task.prompt))
        if take >= len(candidates):
            selected.extend(candidates)
            continue
        bins = [candidates[index::take] for index in range(take)]
        rng = random.Random(f"{seed}:{language}")
        selected.extend(rng.choice(bucket) for bucket in bins)
    return sorted(selected, key=lambda task: task.task_id)


def _task_record(
    task: PolyglotTask,
    model: Any,
    tokenizer: Any,
    *,
    source_root: Path,
    image: str,
    cache_root: Path,
    max_new_tokens: int,
    generator: LocalSGLangGenerator | None = None,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": WHOLE_FILE_SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    turns = []

    draft = generate_turn(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        generator=generator,
    )
    draft_parsed = parse_whole_file_edit(
        draft["completion"],
        task.solution_files,
    )
    draft_verification = verify_edit(
        task,
        draft_parsed,
        source_root=source_root,
        image=image,
        cache_root=cache_root,
    )
    draft.update(
        {
            "turn_index": 0,
            "kind": "draft_edit",
            "parsed_edit": {
                "files": draft_parsed.files,
                "format_compliant": draft_parsed.format_compliant,
                "errors": list(draft_parsed.errors),
            },
            "verification": asdict(draft_verification),
        }
    )
    turns.append(draft)

    if not draft_verification.passed:
        feedback = test_feedback(draft_verification, task.solution_files)
        messages.extend(
            [
                {"role": "assistant", "content": draft["completion"]},
                {"role": "user", "content": feedback},
            ]
        )
        revised = generate_turn(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            generator=generator,
        )
        revised_parsed = parse_whole_file_edit(
            revised["completion"],
            task.solution_files,
        )
        revised_verification = verify_edit(
            task,
            revised_parsed,
            source_root=source_root,
            image=image,
            cache_root=cache_root,
        )
        revised.update(
            {
                "turn_index": 1,
                "kind": "revised_edit",
                "feedback": feedback,
                "parsed_edit": {
                    "files": revised_parsed.files,
                    "format_compliant": revised_parsed.format_compliant,
                    "errors": list(revised_parsed.errors),
                },
                "verification": asdict(revised_verification),
            }
        )
        turns.append(revised)

    final = turns[-1]
    action_span = final["action_char_span_half_open"]
    action = (
        final["completion"][slice(*action_span)]
        if action_span is not None
        else ""
    )
    valid_final_boundary = bool(
        final["stopped_on_eos"] and action_span is not None
    )
    valid_final_action = bool(valid_final_boundary and action.strip())
    return {
        "schema_version": 3,
        "benchmark": "aider_polyglot",
        "task": asdict(task),
        "messages": messages,
        "turns": turns,
        "draft_passed": bool(draft_verification.passed),
        "final_passed": bool(final["verification"]["passed"]),
        "repair_attempted": len(turns) == 2,
        "repair_improved": (
            len(turns) == 2
            and not draft_verification.passed
            and bool(final["verification"]["passed"])
        ),
        "valid_final_boundary": valid_final_boundary,
        "valid_final_action": valid_final_action,
    }


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["task"]["task_id"])
    return ids


def _write_summary(
    path: Path,
    records_path: Path,
    *,
    protocol: dict[str, Any],
) -> None:
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    count = len(records)
    by_language = {}
    for language in sorted({row["task"]["language"] for row in records}):
        rows = [row for row in records if row["task"]["language"] == language]
        by_language[language] = {
            "count": len(rows),
            "draft_pass_rate": sum(row["draft_passed"] for row in rows) / len(rows),
            "final_pass_rate": sum(row["final_passed"] for row in rows) / len(rows),
            "valid_final_action_rate": (
                sum(row["valid_final_action"] for row in rows) / len(rows)
            ),
        }
    summary = {
        "protocol": protocol,
        "task_count": count,
        "draft_passes": sum(row["draft_passed"] for row in records),
        "final_passes": sum(row["final_passed"] for row in records),
        "repair_attempts": sum(row["repair_attempted"] for row in records),
        "repair_improvements": sum(row["repair_improved"] for row in records),
        "format_compliant_turns": sum(
            turn["parsed_edit"]["format_compliant"]
            for row in records
            for turn in row["turns"]
        ),
        "total_turns": sum(len(row["turns"]) for row in records),
        "valid_final_boundaries": sum(
            row["valid_final_boundary"] for row in records
        ),
        "valid_final_actions": sum(row["valid_final_action"] for row in records),
        "languages": by_language,
    }
    summary["draft_pass_rate"] = summary["draft_passes"] / count if count else 0.0
    summary["final_pass_rate"] = summary["final_passes"] / count if count else 0.0
    summary["format_compliance_rate"] = (
        summary["format_compliant_turns"] / summary["total_turns"]
        if summary["total_turns"]
        else 0.0
    )
    summary["valid_final_boundary_rate"] = (
        summary["valid_final_boundaries"] / count if count else 0.0
    )
    summary["valid_final_action_rate"] = (
        summary["valid_final_actions"] / count if count else 0.0
    )
    path.write_text(_json(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/external/aider-polyglot"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image", default="flashtrace-aider-benchmark")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/external/.cache/aider"),
    )
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--sglang-base-url",
        help="Loopback-only SGLang endpoint, for example http://127.0.0.1:8001.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/exp4/output/local_generation/qwen3_8b"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subprocess.run(
        ["docker", "image", "inspect", args.image],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source_commit = _command_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"]
    )
    if source_commit != POLYGLOT_COMMIT:
        raise RuntimeError(
            f"Polyglot source is {source_commit}; expected {POLYGLOT_COMMIT}"
        )
    tasks = discover_tasks(args.source_root)
    selected = _stratified_selection(tasks, args.num_tasks, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        selected,
        args.output_dir / "manifest.jsonl",
        source_root=args.source_root,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )
    generator = (
        LocalSGLangGenerator(args.sglang_base_url, tokenizer)
        if args.sglang_base_url
        else None
    )
    model = None
    if generator is None:
        device_index = int(args.device.split(":")[-1])
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map={"": device_index},
        )
        model.eval()

    protocol = {
        "model": args.model,
        "revision": args.revision,
        "generation_backend": (
            "local_sglang_native_generate"
            if generator is not None
            else "transformers_local"
        ),
        "dtype": "bfloat16",
        "attention": "eager",
        "device": args.device,
        "decoding": "greedy",
        "enable_thinking": True,
        "preserve_thinking": True,
        "max_new_tokens": args.max_new_tokens,
        "source_commit": POLYGLOT_COMMIT,
        "docker_image": args.image,
        "docker_cache_root": str(args.cache_root.resolve()),
        "seed": args.seed,
        "selection": "all" if args.num_tasks is None else "language_and_length_stratified",
    }
    (args.output_dir / "protocol.json").write_text(
        _json(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "provenance.json").write_text(
        _json(
            _provenance(
                args=args,
                source_root=args.source_root,
                generator=generator,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    records_path = args.output_dir / "trajectories.jsonl"
    completed = _existing_ids(records_path)
    with records_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(selected, 1):
            if task.task_id in completed:
                print(f"[{index}/{len(selected)}] skip {task.task_id}", flush=True)
                continue
            print(f"[{index}/{len(selected)}] {task.task_id}", flush=True)
            record = _task_record(
                task,
                model,
                tokenizer,
                source_root=args.source_root,
                image=args.image,
                cache_root=args.cache_root,
                max_new_tokens=args.max_new_tokens,
                generator=generator,
            )
            handle.write(_json(record) + "\n")
            handle.flush()
            print(
                f"  draft={record['draft_passed']} final={record['final_passed']}",
                flush=True,
            )
            _write_summary(
                args.output_dir / "summary.json",
                records_path,
                protocol=protocol,
            )
            if generator is None:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
