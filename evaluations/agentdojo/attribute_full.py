#!/usr/bin/env python3
"""Attribute validator-confirmed target actions from a frozen AgentDojo run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluations.agentdojo.e2e import _attribute_generated_target
from evaluations.agentdojo.pilot import (
    LLMIFRAttribution,
    LLMIFRAttributionBoth,
    _find_injection_span,
)


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible(record: dict[str, Any]) -> bool:
    base_eligible = bool(
        record.get("attack_success")
        and record.get("generated_target_action")
        and record.get("target_event") is not None
    )
    if not base_eligible:
        return False
    try:
        _find_injection_span(str(record["target_event"]["rendered_prompt"]))
    except ValueError:
        return False
    return True


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["case_id"]
        for row in _records(path)
    }


def _synchronize_redacted(raw_path: Path, redacted_path: Path) -> None:
    if not raw_path.exists():
        return
    redacted_ids = _existing_ids(redacted_path)
    missing = [
        row
        for row in _records(raw_path)
        if row["case_id"] not in redacted_ids
    ]
    if not missing:
        return
    with redacted_path.open("a", encoding="utf-8") as handle:
        for row in missing:
            handle.write(_json(_redacted(row)) + "\n")


def _redacted(row: dict[str, Any]) -> dict[str, Any]:
    redacted = {
        "case_id": row["case_id"],
        "suite": row["suite"],
        "realization_stage": row.get("realization_stage"),
        "status": row["status"],
        "error_category": (
            row["error"].split(":", 1)[0] if row.get("error") else None
        ),
    }
    attribution = row.get("attribution")
    if attribution is None:
        return redacted
    redacted["attribution"] = {
        "target_provenance": attribution.get(
            "provenance",
            "actual_autonomous_qwen_generation",
        ),
        "target_teacher_forced": bool(attribution.get("teacher_forced", False)),
        "output_intervention": attribution.get("output_intervention"),
        "prompt_token_count": attribution["prompt_token_count"],
        "gold_injection_token_count": len(
            attribution["gold_injection"]["token_indices"]
        ),
        "generated_target_token_count": attribution["generated_target"][
            "token_count"
        ],
        "target_call_token_span_inclusive": attribution["generated_target"][
            "target_call_token_span_inclusive"
        ],
        "thinking_bridge_token_span_inclusive": attribution["generated_target"][
            "thinking_bridge_token_span_inclusive"
        ],
        "generated_target_token_provenance_counts": attribution[
            "generated_target"
        ].get("token_provenance_counts"),
        "methods": {
            method: {
                "runtime_seconds": values["runtime_seconds"],
                "localization_metrics": values["localization_metrics"],
            }
            for method, values in attribution["methods"].items()
        },
    }
    return redacted


def _summary(
    path: Path,
    *,
    generation_records: int,
    eligible_records: int,
    attacked_inventory: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    rows = list(_records(path)) if path.exists() else []
    successful = [row for row in rows if row["status"] == "success"]
    methods = sorted(
        {
            method
            for row in successful
            for method in row["attribution"]["methods"]
        }
    )
    metrics = (
        "recovery_at_5pct",
        "recovery_at_10pct",
        "recovery_at_20pct",
        "pointing_game",
        "injection_mass_fraction",
    )
    def aggregate_methods(selected: list[dict[str, Any]]) -> dict[str, Any]:
        aggregates = {}
        for method in methods:
            values = [
                row["attribution"]["methods"][method]
                for row in selected
                if method in row["attribution"]["methods"]
            ]
            if not values:
                continue
            method_summary: dict[str, Any] = {"count": len(values)}
            for metric in metrics:
                samples = [
                    float(value["localization_metrics"][metric])
                    for value in values
                ]
                method_summary[f"{metric}_mean"] = mean(samples)
                method_summary[f"{metric}_std"] = (
                    pstdev(samples) if len(samples) > 1 else 0.0
                )
            runtimes = [float(value["runtime_seconds"]) for value in values]
            method_summary["runtime_seconds_mean"] = mean(runtimes)
            method_summary["runtime_seconds_std"] = (
                pstdev(runtimes) if len(runtimes) > 1 else 0.0
            )
            aggregates[method] = method_summary
        return aggregates

    stages = sorted({str(row.get("realization_stage") or "unrecorded") for row in rows})
    return {
        "protocol": protocol,
        "generation_record_count": generation_records,
        "eligible_localization_count": eligible_records,
        "attacked_inventory": attacked_inventory,
        "eligible_coverage_over_attacked_inventory": (
            eligible_records / attacked_inventory
        ),
        "attempted_attribution_count": len(rows),
        "successful_attribution_count": len(successful),
        "failed_attribution_count": len(rows) - len(successful),
        "methods": aggregate_methods(successful),
        "stages": {
            stage: {
                "attempted_attribution_count": sum(
                    str(row.get("realization_stage") or "unrecorded") == stage
                    for row in rows
                ),
                "successful_attribution_count": sum(
                    str(row.get("realization_stage") or "unrecorded") == stage
                    for row in successful
                ),
                "methods": aggregate_methods(
                    [
                        row
                        for row in successful
                        if str(row.get("realization_stage") or "unrecorded")
                        == stage
                    ]
                ),
            }
            for stage in stages
        },
    }


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
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-hops", type=int, default=1)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--attacked-inventory", type=int, default=949)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--sink-chunk-tokens", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/agentdojo/results/full/attribution_qwen3_8b"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attacked_inventory < 1:
        raise ValueError("--attacked-inventory must be positive.")
    generation_records = 0
    eligible_records = 0
    eligible_stage_counts: dict[str, int] = defaultdict(int)
    for record in _records(args.generation_jsonl):
        generation_records += 1
        is_eligible = _eligible(record)
        eligible_records += int(is_eligible)
        if is_eligible:
            eligible_stage_counts[
                str(record.get("realization_stage") or "unrecorded")
            ] += 1
    if eligible_records == 0:
        raise SystemExit("No validator-confirmed generated target actions found.")
    selected_records = (
        eligible_records
        if args.num_examples is None
        else min(args.num_examples, eligible_records)
    )
    if selected_records < 1:
        raise ValueError("--num-examples must be positive.")

    protocol = {
        "generation_jsonl": str(args.generation_jsonl.resolve()),
        "generation_jsonl_sha256": _sha256_file(args.generation_jsonl),
        "attribution_model": args.model,
        "attribution_revision": args.revision,
        "n_hops": args.n_hops,
        "chunk_tokens": args.chunk_tokens,
        "sink_chunk_tokens": args.sink_chunk_tokens,
        "attacked_inventory": args.attacked_inventory,
        "teacher_forced_frozen_generated_trajectory": True,
        "generation_realization_stage_counts": dict(
            sorted(eligible_stage_counts.items())
        ),
        "eligibility": (
            "official_attack_validator_success_exact_generated_target_action_"
            "and_complete_injection_span"
        ),
        "recompute_attention": True,
        "recompute_query_rows_only": True,
        "selection": (
            "all_eligible"
            if args.num_examples is None
            else f"first_{selected_records}_eligible_in_manifest_order"
        ),
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

    model, tokenizer = _load_model(args.model, args.revision, args.device)
    ifr = LLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
        show_progress=False,
        recompute_attention=True,
    )
    flashtrace = LLMIFRAttributionBoth(
        model,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
        show_progress=False,
        recompute_attention=True,
    )

    raw_path = args.output_dir / "attribution.jsonl"
    redacted_path = args.output_dir / "attribution.redacted.jsonl"
    _synchronize_redacted(raw_path, redacted_path)
    completed = _existing_ids(raw_path)
    with (
        raw_path.open("a", encoding="utf-8") as raw_handle,
        redacted_path.open("a", encoding="utf-8") as redacted_handle,
    ):
        current = 0
        for record in _records(args.generation_jsonl):
            if not _eligible(record):
                continue
            current += 1
            if current > selected_records:
                break
            case_id = record["case"]["case_id"]
            if case_id in completed:
                print(f"[{current}/{selected_records}] skip {case_id}", flush=True)
                continue
            print(f"[{current}/{selected_records}] {case_id}", flush=True)
            row = {
                "case_id": case_id,
                "suite": record["case"]["suite"],
                "realization_stage": record.get("realization_stage"),
                "status": "success",
                "error": None,
                "attribution": None,
            }
            try:
                row["attribution"] = _attribute_generated_target(
                    record,
                    tokenizer,
                    ifr,
                    flashtrace,
                    args.n_hops,
                )
            except Exception as exc:  # Keep every failure in the audit denominator.
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            raw_handle.write(_json(row) + "\n")
            raw_handle.flush()
            redacted_handle.write(_json(_redacted(row)) + "\n")
            redacted_handle.flush()
            torch.cuda.empty_cache()
            summary = _summary(
                redacted_path,
                generation_records=generation_records,
                eligible_records=eligible_records,
                attacked_inventory=args.attacked_inventory,
                protocol=protocol,
            )
            (args.output_dir / "summary.json").write_text(
                _json(summary, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
