#!/usr/bin/env python3
"""Run all paper Aider baselines over frozen execution-grounded trajectories."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flashtrace.attribution as flashtrace_attribution
import flashtrace.core as flashtrace_core
import flashtrace.improved as flashtrace_improved
import ft_ifr_improve
import llm_attr
import llm_attr_eval
from exp.exp4.run_exp import (
    _last_meaningful_code_line_token_span,
    _row_faithfulness_scores,
    _trajectory_trace_record,
)
from exp.exp4.trajectory_utils import AiderExample, load_aider


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ATTENTION_IMPLEMENTATION = "sdpa"
METHODS = (
    "ifr_last_output_line",
    "ifr_exhaustive_full_action",
    "attnlrp_full_action",
    "flashtrace_k1_full_action",
)


@contextmanager
def _attnlrp_gradient_checkpointing(model: Any) -> Iterator[None]:
    """Use deterministic layer checkpointing for the AttnLRP backward pass."""

    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError(
            "The attribution model does not support gradient checkpointing."
        )

    was_training = bool(model.training)
    was_checkpointing = bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    if not was_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    try:
        yield
    finally:
        model.train(was_training)
        if not was_checkpointing:
            model.gradient_checkpointing_disable()


class ResponseOnlyAttributionEvaluator(
    llm_attr_eval.LLMAttributionEvaluator
):
    """Score the exact response logits without materializing prompt logits."""

    @torch.no_grad()
    def compute_logprob_response_given_prompt(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        response_start = int(prompt_ids.shape[1])
        response_length = int(response_ids.shape[1])
        positions = torch.arange(
            response_start - 1,
            response_start + response_length - 1,
            device=input_ids.device,
        )
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=positions,
        ).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _existing(path: Path) -> set[tuple[str, str]]:
    return {
        (row["example_id"], row["method"])
        for row in _rows(path)
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "result": (
            None
            if row["result"] is None
            else {
                key: value
                for key, value in row["result"].items()
                if key != "trace"
            }
        ),
    }


def _synchronize_metrics(raw_path: Path, metrics_path: Path) -> None:
    metric_keys = _existing(metrics_path)
    missing = [
        row
        for row in _rows(raw_path)
        if (row["example_id"], row["method"]) not in metric_keys
    ]
    if not missing:
        return
    with metrics_path.open("a", encoding="utf-8") as handle:
        for row in missing:
            handle.write(_json(_compact(row)) + "\n")


def _command_output(command: Sequence[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def _load_model(
    model_id: str,
    revision: str,
    device: str,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    index = int(device.split(":")[-1])
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map={"": index},
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    model.eval()
    return model, tokenizer


def _method_result(
    method: str,
    example: AiderExample,
    *,
    tokenizer: Any,
    ifr: Any,
    attnlrp: Any,
    flashtrace: Any,
    evaluator: Any,
    perturbation_steps: int,
    attnlrp_gradient_checkpointing: bool,
) -> dict[str, Any]:
    if example.sink_span is None:
        raise ValueError("Missing final action token span.")
    action_span = tuple(example.sink_span)
    last_line_span = tuple(
        _last_meaningful_code_line_token_span(tokenizer, example.target)
    )

    started = time.perf_counter()
    if method == "ifr_last_output_line":
        result = ifr.calculate_ifr_for_all_positions_output_only(
            example.prompt,
            target=example.target,
            sink_span=last_line_span,
        )
        explained_span = last_line_span
        attributor = ifr
    elif method == "ifr_exhaustive_full_action":
        result = ifr.calculate_ifr_for_all_positions_output_only(
            example.prompt,
            target=example.target,
            sink_span=action_span,
        )
        explained_span = action_span
        attributor = ifr
    elif method == "attnlrp_full_action":
        checkpoint_context = (
            _attnlrp_gradient_checkpointing(attnlrp.model)
            if attnlrp_gradient_checkpointing
            else nullcontext()
        )
        with checkpoint_context:
            result = attnlrp.calculate_attnlrp_ft_hop0(
                example.prompt,
                target=example.target,
                sink_span=action_span,
                thinking_span=example.thinking_span,
                neg_handling="drop",
                norm_mode="norm",
            )
        explained_span = action_span
        attributor = attnlrp
    elif method == "flashtrace_k1_full_action":
        result = flashtrace.calculate_ifr_multi_hop_both(
            example.prompt,
            target=example.target,
            sink_span=action_span,
            thinking_span=example.thinking_span,
            n_hops=1,
        )
        explained_span = action_span
        attributor = flashtrace
    else:
        raise ValueError(f"Unknown method: {method}")
    attribution_seconds = time.perf_counter() - started

    row = result.get_all_token_attrs(list(explained_span))[1]
    prompt_length = len(result.prompt_tokens)
    target_length = len(result.generation_tokens)
    prompt_row = row[:, :prompt_length]
    keep = ft_ifr_improve.keep_token_indices(list(result.prompt_tokens))
    trace = _trajectory_trace_record(
        example_idx=0,
        example=example,
        method=method,
        sink=(
            "last_output_line"
            if method == "ifr_last_output_line"
            else "full_action"
        ),
        tokenizer=tokenizer,
        attribution_result=result,
        row_vector=row,
    )
    del result
    gc.collect()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    rise, mas = _row_faithfulness_scores(
        llm_evaluator=evaluator,
        attribution_prompt=prompt_row,
        prompt=example.prompt,
        generation=example.target,
        user_prompt_indices=list(attributor.user_prompt_indices),
        keep_prompt_token_indices=keep,
        score_generation_span=action_span,
        k=perturbation_steps,
    )
    faithfulness_seconds = time.perf_counter() - started
    return {
        "explained_target_token_span_inclusive": list(explained_span),
        "scored_target_token_span_inclusive": list(action_span),
        "prompt_token_count": prompt_length,
        "target_token_count_with_eos": target_length,
        "rise": rise,
        "mas": mas,
        "attribution_seconds": attribution_seconds,
        "faithfulness_seconds": faithfulness_seconds,
        "total_seconds": attribution_seconds + faithfulness_seconds,
        "source_segments": trace["segments"],
        "unassigned_source_fraction": trace["unassigned_fraction"],
        "trace": trace,
    }


def _summary(
    path: Path,
    *,
    protocol: dict[str, Any],
    expected_examples: int,
) -> dict[str, Any]:
    rows = list(_rows(path))
    summary: dict[str, Any] = {
        "protocol": protocol,
        "expected_example_count": expected_examples,
        "attempted_method_runs": len(rows),
        "methods": {},
    }
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        successful = [row for row in method_rows if row["status"] == "success"]
        method_summary: dict[str, Any] = {
            "attempted": len(method_rows),
            "successful": len(successful),
            "failed": len(method_rows) - len(successful),
        }
        for subset_name, subset in (
            ("all_valid", successful),
            (
                "execution_success",
                [row for row in successful if row["final_passed"]],
            ),
        ):
            values: dict[str, Any] = {"count": len(subset)}
            for metric in (
                "rise",
                "mas",
                "attribution_seconds",
                "faithfulness_seconds",
                "total_seconds",
            ):
                samples = [float(row["result"][metric]) for row in subset]
                values[f"{metric}_mean"] = mean(samples) if samples else None
                values[f"{metric}_std"] = (
                    pstdev(samples) if len(samples) > 1 else 0.0 if samples else None
                )
            method_summary[subset_name] = values
        summary["methods"][method] = method_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument(
        "--example-id",
        action="append",
        default=None,
        help="Restrict attribution to one or more exact task IDs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--sink-chunk-tokens", type=int, default=32)
    parser.add_argument("--perturbation-steps", type=int, default=20)
    parser.add_argument(
        "--attnlrp-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/exp4/output/local_attribution/qwen3_8b"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = _load_model(args.model, args.revision, args.device)
    examples = [
        example
        for example in load_aider(args.generation_jsonl, tokenizer=tokenizer)
        if example.metadata.get("valid_final_action")
    ]
    if args.example_id:
        requested = set(args.example_id)
        examples = [
            example
            for example in examples
            if str(example.metadata["example_id"]) in requested
        ]
        found = {
            str(example.metadata["example_id"])
            for example in examples
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested example IDs are unavailable: {missing}")
    if args.num_examples is not None:
        examples = examples[: args.num_examples]
    if not examples:
        raise SystemExit("No valid frozen Aider actions found.")

    protocol = {
        "generation_jsonl": str(args.generation_jsonl.resolve()),
        "generation_jsonl_sha256": _sha256_file(args.generation_jsonl),
        "selected_example_ids": (
            sorted(args.example_id) if args.example_id else None
        ),
        "attribution_model": args.model,
        "attribution_revision": args.revision,
        "methods": list(args.methods),
        "chunk_tokens": args.chunk_tokens,
        "sink_chunk_tokens": args.sink_chunk_tokens,
        "flashtrace_hops": 1,
        "attnlrp_neg_handling": "drop",
        "attnlrp_norm_mode": "norm",
        "attnlrp_gradient_checkpointing": (
            args.attnlrp_gradient_checkpointing
        ),
        "attnlrp_checkpointing_use_reentrant": False,
        "attribution_model_attention_dropout": float(
            getattr(model.config, "attention_dropout", 0.0)
        ),
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "ifr_recompute_attention": True,
        "ifr_recompute_query_rows_only": True,
        "faithfulness_scored_span": "complete_final_edit_action_only",
        "faithfulness_logits": "response_prediction_positions_only",
        "faithfulness_use_cache": False,
        "perturbation_steps": args.perturbation_steps,
        "teacher_forced_frozen_generated_trajectory": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(
        _json(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "flashtrace_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "flashtrace_dirty": bool(
            _command_output(["git", "status", "--short"])
        ),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "flashtrace_attribution_sha256": hashlib.sha256(
            Path(flashtrace_attribution.__file__).read_bytes()
        ).hexdigest(),
        "flashtrace_core_sha256": hashlib.sha256(
            Path(flashtrace_core.__file__).read_bytes()
        ).hexdigest(),
        "flashtrace_improved_sha256": hashlib.sha256(
            Path(flashtrace_improved.__file__).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(int(args.device.split(":")[-1])),
    }
    (args.output_dir / "provenance.json").write_text(
        _json(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    evaluator = ResponseOnlyAttributionEvaluator(model, tokenizer)
    ifr = llm_attr.LLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
        show_progress=False,
        recompute_attention=True,
    )
    attnlrp = llm_attr.LLMLRPAttribution(model, tokenizer)
    flashtrace = ft_ifr_improve.LLMIFRAttributionBoth(
        model,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
        show_progress=False,
        recompute_attention=True,
    )

    output_path = args.output_dir / "attribution.jsonl"
    metrics_path = args.output_dir / "metrics.jsonl"
    _synchronize_metrics(output_path, metrics_path)
    completed = _existing(output_path)
    total_runs = len(examples) * len(args.methods)
    with (
        output_path.open("a", encoding="utf-8") as handle,
        metrics_path.open("a", encoding="utf-8") as metrics_handle,
    ):
        run_index = 0
        for example in examples:
            example_id = str(example.metadata["example_id"])
            for method in args.methods:
                run_index += 1
                if (example_id, method) in completed:
                    print(f"[{run_index}/{total_runs}] skip {example_id} {method}")
                    continue
                print(f"[{run_index}/{total_runs}] {example_id} {method}", flush=True)
                row = {
                    "example_id": example_id,
                    "language": example.metadata["language"],
                    "method": method,
                    "status": "success",
                    "error": None,
                    "final_passed": bool(example.metadata["final_passed"]),
                    "repair_attempted": bool(
                        example.metadata["repair_attempted"]
                    ),
                    "result": None,
                }
                try:
                    row["result"] = _method_result(
                        method,
                        example,
                        tokenizer=tokenizer,
                        ifr=ifr,
                        attnlrp=attnlrp,
                        flashtrace=flashtrace,
                        evaluator=evaluator,
                        perturbation_steps=args.perturbation_steps,
                        attnlrp_gradient_checkpointing=(
                            args.attnlrp_gradient_checkpointing
                        ),
                    )
                except Exception as exc:  # Preserve every failure in the audit.
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                handle.write(_json(row) + "\n")
                handle.flush()
                compact = _compact(row)
                metrics_handle.write(_json(compact) + "\n")
                metrics_handle.flush()
                gc.collect()
                torch.cuda.empty_cache()
                summary = _summary(
                    metrics_path,
                    protocol=protocol,
                    expected_examples=len(examples),
                )
                (args.output_dir / "summary.json").write_text(
                    _json(summary, indent=2) + "\n",
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
