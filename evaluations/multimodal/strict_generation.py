"""Generate strict visual-reasoning model records with Qwen3-VL-Thinking.

The dataset manifest, model output, and generation evaluation are written to
three separate files:

* dataset JSONL: image/question plus reference and evidence metadata;
* model JSONL: image/question plus the model's own THINKING and whole OUTPUT;
* evaluation JSONL: correctness, stability, and image-dependence checks.

Dataset rationales and functional programs are never included in the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


SCHEMA_VERSION = 2
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_PROMPT_PROFILE = "concise"
PROMPT_TEMPLATES = {
    "concise": """Answer the visual question using the image. Think carefully through the visual evidence before giving the final answer. After the reasoning, give one concise final answer and do not add commentary after it.

Question: {question}""",
    "long_form": """Answer the visual question using the image. Think carefully through the visual evidence before giving the final answer. After the reasoning, give a detailed, self-contained final answer. Include only claims supported by the image, state uncertainty when the image is insufficient, and do not add commentary after the final answer.

Question: {question}""",
}
# Backward-compatible alias for earlier strict artifacts.
DEFAULT_PROMPT = PROMPT_TEMPLATES[DEFAULT_PROMPT_PROFILE]


def render_prompt(question: str, profile: str = DEFAULT_PROMPT_PROFILE) -> str:
    try:
        template = PROMPT_TEMPLATES[profile]
    except KeyError as error:
        raise ValueError(f"unknown prompt profile: {profile!r}") from error
    return template.format(question=question)


def model_record_prompt(record: Mapping[str, Any]) -> str:
    metadata = record.get("generation_metadata") or {}
    profile = str(metadata.get("prompt_profile", DEFAULT_PROMPT_PROFILE))
    return render_prompt(str(record["I_QUESTION"]), profile)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def split_thinking_output(response: str) -> tuple[str, str, tuple[int, int], tuple[int, int]]:
    """Split a Qwen Thinking response without modifying the final OUTPUT.

    Returned character spans use Python's half-open convention.  ``OUTPUT`` is
    the complete non-whitespace tail after ``</think>``; labels and markup are
    deliberately preserved because the output itself is the final answer.
    """

    marker = "</think>"
    marker_start = response.find(marker)
    if marker_start < 0:
        raise ValueError("strict Thinking response has no </think> terminator")

    thinking_start = 0
    opening = "<think>"
    if response.startswith(opening):
        thinking_start = len(opening)
    while thinking_start < marker_start and response[thinking_start].isspace():
        thinking_start += 1
    thinking_end = marker_start
    while thinking_end > thinking_start and response[thinking_end - 1].isspace():
        thinking_end -= 1

    output_start = marker_start + len(marker)
    while output_start < len(response) and response[output_start].isspace():
        output_start += 1
    output_end = len(response)
    while output_end > output_start and response[output_end - 1].isspace():
        output_end -= 1

    if thinking_end <= thinking_start:
        raise ValueError("strict Thinking response has empty THINKING")
    if output_end <= output_start:
        raise ValueError("strict Thinking response has empty OUTPUT")
    return (
        response[thinking_start:thinking_end],
        response[output_start:output_end],
        (thinking_start, thinking_end),
        (output_start, output_end),
    )


def char_span_to_token_span(
    tokenizer: Any,
    text: str,
    char_span: tuple[int, int],
) -> tuple[int, int]:
    """Map a half-open character span to an inclusive generated-token span."""

    start, end = char_span
    if not 0 <= start < end <= len(text):
        raise ValueError(f"invalid character span {char_span} for {len(text)} characters")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], Sequence) and len(offsets[0]) == 2:
        pass
    elif offsets and isinstance(offsets[0], Sequence):
        offsets = offsets[0]
    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if int(token_end) > start and int(token_start) < end
    ]
    if not overlapping:
        raise ValueError(f"character span {char_span} has no tokenizer overlap")
    return overlapping[0], overlapping[-1]


def validate_model_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "benchmark",
        "sample_id",
        "I_IMAGE",
        "I_QUESTION",
        "THINKING",
        "OUTPUT",
        "THINKING_SPAN",
        "OUTPUT_SPAN",
        "raw_response",
        "model",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"model record is missing keys: {sorted(missing)}")
    forbidden = {
        "REFERENCE_OUTPUT",
        "EVIDENCE_BOXES",
        "EVIDENCE_MASK",
        "EVIDENCE_MASKS",
        "functional_program",
        "human_rationale",
    }
    leaked = forbidden.intersection(record)
    if leaked:
        raise ValueError(f"evaluation fields leaked into model record: {sorted(leaked)}")
    if not str(record["THINKING"]).strip() or not str(record["OUTPUT"]).strip():
        raise ValueError("THINKING and OUTPUT must be non-empty model generations")
    for name in ("THINKING_SPAN", "OUTPUT_SPAN"):
        span = record[name]
        if not isinstance(span, Sequence) or len(span) != 2:
            raise ValueError(f"{name} must be an inclusive two-token span")
        if int(span[0]) < 0 or int(span[1]) < int(span[0]):
            raise ValueError(f"{name} is invalid: {span}")
    generation_metadata = record.get("generation_metadata") or {}
    generated_ids = generation_metadata.get("original_generated_token_ids")
    teacher_forced_ids = generation_metadata.get("teacher_forced_token_ids")
    if (
        generated_ids is not None
        and teacher_forced_ids is not None
        and generated_ids != teacher_forced_ids
    ):
        raise ValueError(
            "generated token IDs differ from decode/re-encoded teacher-forced IDs"
        )


def _messages(prompt: str, image: Image.Image | Sequence[Image.Image]) -> list[dict[str, Any]]:
    images = (
        list(image)
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray))
        else [image]
    )
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": item} for item in images),
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _model_inputs(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
) -> dict[str, Any]:
    inputs = processor.apply_chat_template(
        _messages(prompt, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    return dict(inputs)


@torch.inference_mode()
def generate_response(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    inputs = _model_inputs(model, processor, image, prompt)
    prompt_length = int(inputs["input_ids"].shape[1])
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    sequences = output.sequences if hasattr(output, "sequences") else output
    generated_ids = sequences[0, prompt_length:]
    eos_token_id = processor.tokenizer.eos_token_id
    if (
        generated_ids.numel()
        and eos_token_id is not None
        and int(generated_ids[-1]) == int(eos_token_id)
    ):
        generated_ids = generated_ids[:-1]
    response = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return response, [int(token_id) for token_id in generated_ids.tolist()]


@torch.inference_mode()
def output_mean_logprob(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str,
    output_span: tuple[int, int],
) -> float:
    """Teacher-force a frozen response and score only its OUTPUT token span."""

    inputs = _model_inputs(model, processor, image, prompt)
    response_ids = processor.tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    output_start, output_end = output_span
    if not 0 <= output_start <= output_end < int(response_ids.shape[1]):
        raise ValueError(
            f"OUTPUT_SPAN {output_span} is outside {response_ids.shape[1]} response tokens"
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    full_ids = torch.cat((inputs["input_ids"], response_ids), dim=1)
    full_mask = torch.cat(
        (
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            torch.ones_like(response_ids),
        ),
        dim=1,
    )
    forward_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }
    token_types = forward_inputs.get("mm_token_type_ids")
    if torch.is_tensor(token_types) and token_types.shape[-1] != full_ids.shape[-1]:
        forward_inputs["mm_token_type_ids"] = F.pad(
            token_types,
            (0, full_ids.shape[-1] - token_types.shape[-1]),
            value=0,
        )
    output = model(input_ids=full_ids, attention_mask=full_mask, **forward_inputs)
    logits = output.logits[
        :,
        prompt_length - 1 : prompt_length - 1 + response_ids.shape[1],
        :,
    ].float()
    selected = logits.log_softmax(dim=-1).gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)
    return float(selected[:, output_start : output_end + 1].mean().item())


_ANSWER_PREFIX = re.compile(
    r"(?is)^\s*(?:<answer>\s*)?(?:\*\*)?"
    r"(?:final\s+answer|answer)\s*:\s*(?:\*\*)?\s*"
)


def normalized_output(output: str) -> str:
    """Normalize only for correctness; the saved OUTPUT remains untouched."""

    value = _ANSWER_PREFIX.sub("", output)
    value = re.sub(r"(?is)</answer>\s*$", "", value)
    value = value.strip().strip("`*_#")
    value = re.sub(r"\s+", " ", value).casefold()
    value = re.sub(r"^[\"']|[\"']$", "", value)
    value = value.rstrip(" .,:;!?")
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return number_words.get(value, value)


def output_correct(output: str, reference: str, benchmark: str) -> bool:
    prediction = normalized_output(output)
    target = normalized_output(reference)
    if benchmark == "clevr_xai_complex":
        boolean_alias = {"yes": "true", "no": "false"}
        prediction = boolean_alias.get(prediction, prediction)
        target = boolean_alias.get(target, target)
    return prediction == target


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolved_revision(model: Any, processor: Any, requested: str | None) -> str:
    candidates = [
        getattr(getattr(model, "config", None), "_commit_hash", None),
        getattr(getattr(processor, "tokenizer", None), "_commit_hash", None),
        requested,
    ]
    return next((str(value) for value in candidates if value), "unknown")


def evaluate_record(
    dataset_record: Mapping[str, Any],
    *,
    model: Any,
    processor: Any,
    model_name: str,
    requested_revision: str | None,
    max_new_tokens: int,
    stability_repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = dataset_record["input"]
    image = Image.open(inputs["I_IMAGE"]).convert("RGB")
    evaluation_metadata = dataset_record.get("evaluation", {}).get("metadata") or {}
    prompt_profile = str(
        evaluation_metadata.get("prompt_profile", DEFAULT_PROMPT_PROFILE)
    )
    prompt_template = PROMPT_TEMPLATES.get(prompt_profile)
    if prompt_template is None:
        raise ValueError(f"unknown prompt profile: {prompt_profile!r}")
    prompt = render_prompt(str(inputs["I_QUESTION"]), prompt_profile)

    started = time.perf_counter()
    response, generated_ids = generate_response(
        model,
        processor,
        image,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    generation_seconds = time.perf_counter() - started
    try:
        thinking, output, thinking_chars, output_chars = split_thinking_output(response)
    except ValueError as error:
        raise ValueError(
            f"{error}; generated_tokens={len(generated_ids)}; "
            f"response_tail={response[-160:]!r}"
        ) from error
    thinking_span = char_span_to_token_span(
        processor.tokenizer, response, thinking_chars
    )
    output_span = char_span_to_token_span(processor.tokenizer, response, output_chars)
    teacher_forced_ids = [
        int(token_id)
        for token_id in processor.tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]
    ]
    if output_span[1] >= len(teacher_forced_ids):
        raise ValueError(
            f"OUTPUT_SPAN {output_span} exceeds {len(teacher_forced_ids)} "
            "teacher-forced tokens"
        )
    token_identity_stable = generated_ids == teacher_forced_ids

    repeated_responses = [response]
    for _ in range(max(1, stability_repeats) - 1):
        repeated, _ = generate_response(
            model,
            processor,
            image,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        repeated_responses.append(repeated)
    stable = all(item == response for item in repeated_responses)

    original_logprob = output_mean_logprob(
        model,
        processor,
        image,
        prompt,
        response,
        output_span,
    )
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(image.size) / 12))
    blurred_logprob = output_mean_logprob(
        model,
        processor,
        blurred,
        prompt,
        response,
        output_span,
    )
    dependence_delta = original_logprob - blurred_logprob
    model_record = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": dataset_record["benchmark"],
        "sample_id": dataset_record["sample_id"],
        "I_IMAGE": inputs["I_IMAGE"],
        "I_QUESTION": inputs["I_QUESTION"],
        "THINKING": thinking,
        "OUTPUT": output,
        "THINKING_SPAN": list(thinking_span),
        "OUTPUT_SPAN": list(output_span),
        "raw_response": response,
        "model": {
            "repo_id": model_name,
            "requested_revision": requested_revision,
            "resolved_revision": _resolved_revision(
                model, processor, requested_revision
            ),
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__,
            "prompt_template_sha256": _sha256_text(prompt_template),
            "chat_template_sha256": _sha256_text(
                str(getattr(processor.tokenizer, "chat_template", ""))
            ),
            "generation": {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "prompt_profile": prompt_profile,
            },
        },
        "generation_metadata": {
            "prompt_profile": prompt_profile,
            "original_generated_token_ids": generated_ids,
            "original_generated_tokens_without_eos": len(generated_ids),
            "teacher_forced_token_ids": teacher_forced_ids,
            "teacher_forced_tokens": len(teacher_forced_ids),
            "generated_teacher_forced_ids_match": token_identity_stable,
            "thinking_tokens": thinking_span[1] - thinking_span[0] + 1,
            "output_tokens": output_span[1] - output_span[0] + 1,
            "seconds": generation_seconds,
        },
    }
    validate_model_record(model_record)
    reference = str(dataset_record["evaluation"]["REFERENCE_OUTPUT"])
    evaluation_record = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": dataset_record["benchmark"],
        "sample_id": dataset_record["sample_id"],
        "REFERENCE_OUTPUT": reference,
        "normalized_prediction": normalized_output(output),
        "normalized_reference": normalized_output(reference),
        "output_correct": output_correct(
            output, reference, str(dataset_record["benchmark"])
        ),
        "generation_stable": stable,
        "stability_repeats": max(1, stability_repeats),
        "original_output_mean_logprob": original_logprob,
        "blurred_output_mean_logprob": blurred_logprob,
        "image_dependence_delta": dependence_delta,
        "image_dependent": dependence_delta > 0.0,
        "generated_teacher_forced_ids_match": token_identity_stable,
        "strict_eligible": (
            output_correct(output, reference, str(dataset_record["benchmark"]))
            and stable
            and dependence_delta > 0.0
            and token_identity_stable
        ),
    }
    return model_record, evaluation_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--stability-repeats", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Run only the named sample ID; may be repeated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed model records and retry only samples without one.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    records = read_jsonl(args.dataset_manifest)
    if args.sample_id:
        requested_ids = set(args.sample_id)
        records = [record for record in records if record["sample_id"] in requested_ids]
        missing = requested_ids.difference(record["sample_id"] for record in records)
        if missing:
            raise ValueError(f"sample IDs not found in manifest: {sorted(missing)}")
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        device_map={"": args.device},
    )
    model.eval()

    model_by_id = {
        record["sample_id"]: record
        for record in (
            read_jsonl(args.model_output)
            if args.resume and args.model_output.exists()
            else []
        )
    }
    evaluation_by_id = {
        record["sample_id"]: record
        for record in (
            read_jsonl(args.evaluation_output)
            if args.resume and args.evaluation_output.exists()
            else []
        )
    }
    record_order = [record["sample_id"] for record in records]
    for index, dataset_record in enumerate(records):
        sample_id = dataset_record["sample_id"]
        if sample_id in model_by_id:
            print(
                f"[{index + 1}/{len(records)}] {sample_id} resume=skip-complete",
                flush=True,
            )
            continue
        try:
            model_record, evaluation_record = evaluate_record(
                dataset_record,
                model=model,
                processor=processor,
                model_name=args.model,
                requested_revision=args.revision,
                max_new_tokens=args.max_new_tokens,
                stability_repeats=args.stability_repeats,
            )
        except Exception as error:
            evaluation_record = {
                "schema_version": SCHEMA_VERSION,
                "benchmark": dataset_record.get("benchmark"),
                "sample_id": dataset_record.get("sample_id"),
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "strict_eligible": False,
            }
            print(
                f"[{index + 1}/{len(records)}] {dataset_record.get('sample_id')} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
        else:
            model_by_id[sample_id] = model_record
            print(
                f"[{index + 1}/{len(records)}] {model_record['sample_id']} "
                f"thinking={model_record['generation_metadata']['thinking_tokens']} "
                f"output={model_record['OUTPUT']!r} "
                f"eligible={evaluation_record['strict_eligible']}",
                flush=True,
            )
        evaluation_by_id[sample_id] = evaluation_record
        write_jsonl(
            [model_by_id[item] for item in record_order if item in model_by_id],
            args.model_output,
        )
        write_jsonl(
            [
                evaluation_by_id[item]
                for item in record_order
                if item in evaluation_by_id
            ],
            args.evaluation_output,
        )

    eligible = sum(
        bool(record.get("strict_eligible")) for record in evaluation_by_id.values()
    )
    print(
        json.dumps(
            {
                "dataset_manifest": str(args.dataset_manifest),
                "model_output": str(args.model_output),
                "evaluation_output": str(args.evaluation_output),
                "model_records": len(model_by_id),
                "strict_eligible": eligible,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
