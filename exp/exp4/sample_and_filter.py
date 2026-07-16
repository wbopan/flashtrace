#!/usr/bin/env python3
"""Construct judge-filtered multi-turn Aider trajectories through a chat API.

Each seed is a legacy exp4 ``input``/``output`` pair.  The generator first
produces an edit, then receives reference-guided *test-style* feedback from the
judge and repairs the edit in later turns.  Only trajectories whose final edit
is judged semantically consistent with the reference output are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from tqdm import tqdm


GEN_SYSTEM_PROMPT = (
    "You are the coding model inside Aider. Solve the user's code-editing task. "
    "Return only the requested file edit in the same filename-and-code-fence style as the task; "
    "do not add commentary outside the edit. Preserve public function and class names."
)

FEEDBACK_SYSTEM_PROMPT = (
    "You simulate Aider's test runner feedback for a coding benchmark. Compare the candidate edit "
    "with the reference solution and identify the smallest concrete correctness issue to fix. "
    "Write concise test-failure-style feedback without quoting or revealing the reference solution. "
    "If the candidate is already correct, request a careful verification/refinement that preserves behavior."
)

JUDGE_SYSTEM_PROMPT = (
    "You judge Aider benchmark edits. Decide whether the candidate implements the same required behavior "
    "as the reference solution and preserves the expected edit format. Reply strictly with True or False."
)


class RateLimitError(RuntimeError):
    """HTTP 429 with a server- or caller-suggested retry delay."""

    def __init__(self, wait_seconds: float, detail: str) -> None:
        super().__init__(detail)
        self.wait_seconds = float(wait_seconds)


@dataclass(frozen=True)
class AiderSeed:
    example_id: str
    prompt: str
    reference_output: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class TrajectoryConfig:
    generator_model: str
    judge_model: str
    assistant_turns: int = 2
    generator_max_tokens: int = 8192
    generator_temperature: float = 0.0


Requester = Callable[..., str]


def call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    timeout: int,
    max_tokens: int,
    temperature: float,
    cache_ttl: int,
    cache_namespace: Optional[str],
    rate_limit_delay: float,
) -> str:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""

    url = api_base.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if cache_ttl > 0:
        cache: Dict[str, Any] = {"ttl": int(cache_ttl)}
        if cache_namespace:
            cache["namespace"] = cache_namespace
        payload["cache"] = cache

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=int(timeout)) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        if exc.code == 429:
            retry_after: Optional[float] = None
            if exc.headers:
                value = exc.headers.get("Retry-After")
                if value:
                    try:
                        retry_after = float(value)
                    except ValueError:
                        retry_after = None
            raise RateLimitError(retry_after or rate_limit_delay, f"API HTTP 429: {detail}") from exc
        raise RuntimeError(f"API HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc

    try:
        decoded = json.loads(response_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode API response: {response_bytes!r}") from exc
    choices = decoded.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty choices from API: {decoded}")
    content = choices[0].get("message", {}).get("content", "")
    if not str(content).strip():
        raise RuntimeError(f"Empty content from API: {decoded}")
    return str(content).strip()


def call_with_retries(
    fn: Callable[[], str],
    *,
    retries: int,
    retry_delay: float,
) -> str:
    """Run one API request with exp2-compatible retry semantics."""

    for attempt in range(int(retries) + 1):
        try:
            return fn()
        except RateLimitError as exc:
            if attempt >= int(retries):
                raise
            time.sleep(exc.wait_seconds)
        except Exception:  # noqa: BLE001 - retry arbitrary transport/provider failures
            if attempt >= int(retries):
                raise
            time.sleep(float(retry_delay))
    raise AssertionError("unreachable")


def parse_bool(text: str) -> bool:
    if not text.strip():
        raise ValueError("Cannot parse an empty boolean judge response.")
    first_line = text.strip().splitlines()[0].strip().lower()
    if first_line in {"true", "yes"}:
        return True
    if first_line in {"false", "no"}:
        return False
    if "true" in first_line and "false" not in first_line:
        return True
    if "false" in first_line:
        return False
    raise ValueError(f"Cannot parse boolean judge response: {text!r}")


def _stable_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_seeds(path: Path) -> List[AiderSeed]:
    seeds: List[AiderSeed] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("input") or row.get("prompt") or "")
            reference = str(row.get("output") or row.get("target") or "")
            if not prompt.strip() or not reference.strip():
                raise ValueError(
                    f"Seed row {row_index} requires non-empty input/output (or prompt/target)."
                )
            metadata = dict(row.get("metadata") or {})
            if row.get("length") is not None:
                metadata.setdefault("length", row.get("length"))
            example_id = str(row.get("id") or metadata.get("example_id") or _stable_id(prompt))
            seeds.append(
                AiderSeed(
                    example_id=example_id,
                    prompt=prompt,
                    reference_output=reference,
                    metadata=metadata,
                )
            )
    return seeds


def build_feedback_messages(seed: AiderSeed, candidate: str) -> List[Dict[str, str]]:
    user = (
        "Aider task:\n"
        f"{seed.prompt}\n\n"
        "Reference edit (private; do not quote it):\n"
        f"{seed.reference_output}\n\n"
        "Candidate edit:\n"
        f"{candidate}\n\n"
        "Return only concise simulated test feedback for the next repair turn."
    )
    return [
        {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_judge_messages(seed: AiderSeed, candidate: str) -> List[Dict[str, str]]:
    user = (
        "Aider task:\n"
        f"{seed.prompt}\n\n"
        "Reference edit:\n"
        f"{seed.reference_output}\n\n"
        "Candidate final edit:\n"
        f"{candidate}\n\n"
        "Output only True if the candidate is semantically correct and usable; otherwise output False."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def feedback_user_message(feedback: str) -> str:
    return (
        "Aider test feedback from the previous edit:\n"
        f"{feedback.strip()}\n\n"
        "Fix the edit to resolve this feedback. Return only the updated file edit in the original format."
    )


def generate_trajectory(
    seed: AiderSeed,
    config: TrajectoryConfig,
    requester: Requester,
) -> tuple[List[Dict[str, str]], str]:
    """Generate a fixed-depth repair trajectory and return its final judge text.

    ``requester`` is injected so the orchestration can be tested without network
    access. It receives ``purpose``, ``model``, ``messages``, ``max_tokens`` and
    ``temperature`` keyword arguments.
    """

    if int(config.assistant_turns) < 2:
        raise ValueError("assistant_turns must be at least 2 for a multi-turn benchmark.")

    stored_messages: List[Dict[str, str]] = [
        {"role": "system", "content": GEN_SYSTEM_PROMPT, "kind": "agent_instruction"},
        {"role": "user", "content": seed.prompt, "kind": "task"},
    ]
    api_messages: List[Dict[str, str]] = [
        {"role": message["role"], "content": message["content"]}
        for message in stored_messages
    ]

    final_candidate = ""
    for turn_index in range(int(config.assistant_turns)):
        final_candidate = requester(
            purpose="generation",
            model=config.generator_model,
            messages=list(api_messages),
            max_tokens=int(config.generator_max_tokens),
            temperature=float(config.generator_temperature),
        ).strip()
        if not final_candidate:
            raise ValueError(f"Generator returned an empty edit at assistant turn {turn_index + 1}.")
        assistant_message = {
            "role": "assistant",
            "content": final_candidate,
            "kind": "draft_edit" if turn_index == 0 else "revised_edit",
        }
        stored_messages.append(assistant_message)
        api_messages.append({"role": "assistant", "content": final_candidate})

        if turn_index >= int(config.assistant_turns) - 1:
            break
        feedback = requester(
            purpose="feedback",
            model=config.judge_model,
            messages=build_feedback_messages(seed, final_candidate),
            max_tokens=512,
            temperature=0.0,
        ).strip()
        if not feedback:
            raise ValueError(f"Judge returned empty feedback after assistant turn {turn_index + 1}.")
        follow_up = feedback_user_message(feedback)
        stored_messages.append({"role": "user", "content": follow_up, "kind": "test_feedback"})
        api_messages.append({"role": "user", "content": follow_up})

    judge_response = requester(
        purpose="judge",
        model=config.judge_model,
        messages=build_judge_messages(seed, final_candidate),
        max_tokens=16,
        temperature=0.0,
    )
    return stored_messages, judge_response


def trajectory_row(
    seed: AiderSeed,
    messages: List[Dict[str, str]],
    judge_response: str,
    config: TrajectoryConfig,
) -> Dict[str, Any]:
    metadata = dict(seed.metadata)
    metadata.update(
        {
            "source": "exp4_aider_legacy_seed",
            "generator_model": config.generator_model,
            "judge_model": config.judge_model,
            "assistant_turns": int(config.assistant_turns),
            "generator_max_tokens": int(config.generator_max_tokens),
            "generator_temperature": float(config.generator_temperature),
            "judge_response": judge_response.strip(),
            "prompt_sha256": hashlib.sha256(seed.prompt.encode("utf-8")).hexdigest(),
            "reference_output_sha256": hashlib.sha256(seed.reference_output.encode("utf-8")).hexdigest(),
        }
    )
    return {
        "schema_version": 2,
        "benchmark": "aider_multiturn",
        "id": seed.example_id,
        "messages": messages,
        "metadata": metadata,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser("Construct judge-filtered multi-turn Aider trajectories.")
    parser.add_argument("--seed_path", type=str, default="exp/exp4/data/aider.jsonl")
    parser.add_argument("--out", type=str, default="exp/exp4/data/aider_multiturn.jsonl")
    parser.add_argument("--max_examples", type=int, default=100, help="Number of judge=True trajectories to retain.")
    parser.add_argument("--assistant_turns", type=int, default=2, help="Number of assistant edit attempts (minimum 2).")
    parser.add_argument("--api_base", type=str, default="http://localhost:4000/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--generator_model", type=str, default="qwen3-235b-a22b-2507")
    parser.add_argument("--judge_model", type=str, default="deepseek-v3-1-terminus")
    parser.add_argument("--api_timeout", type=int, default=300)
    parser.add_argument("--api_max_tokens", type=int, default=8192)
    parser.add_argument("--api_temperature", type=float, default=0.0)
    parser.add_argument("--api_cache_ttl", type=int, default=600)
    parser.add_argument("--api_cache_namespace", type=str, default="flashtrace-exp4-aider-agent")
    parser.add_argument("--retries", type=int, default=2, help="Additional retries for each API request.")
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--rate_limit_delay", type=float, default=5.0)
    parser.add_argument("--request_interval", type=float, default=1.0)
    parser.add_argument("--judge_interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.assistant_turns < 2:
        raise SystemExit("--assistant_turns must be at least 2.")
    seed_path = Path(args.seed_path)
    if not seed_path.exists():
        raise SystemExit(f"Missing legacy Aider seed JSONL: {seed_path}")
    api_key = args.api_key or os.environ.get("FLASHTRACE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set --api_key or FLASHTRACE_API_KEY/OPENAI_API_KEY for API access.")

    seeds = load_seeds(seed_path)
    config = TrajectoryConfig(
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        assistant_turns=args.assistant_turns,
        generator_max_tokens=args.api_max_tokens,
        generator_temperature=args.api_temperature,
    )

    def requester(*, purpose: str, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        response = call_with_retries(
            lambda: call_chat_api(
                args.api_base,
                api_key,
                model,
                messages,
                timeout=args.api_timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                cache_ttl=args.api_cache_ttl,
                cache_namespace=args.api_cache_namespace,
                rate_limit_delay=args.rate_limit_delay,
            ),
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        interval = args.request_interval if purpose == "generation" else args.judge_interval
        if interval > 0:
            time.sleep(interval)
        return response

    kept: List[Dict[str, Any]] = []
    attempted = 0
    progress = tqdm(total=int(args.max_examples), desc="Kept (judge=True)")
    for attempted, seed in enumerate(tqdm(seeds, desc="Aider seeds"), 1):
        if len(kept) >= int(args.max_examples):
            break
        messages, judge_response = generate_trajectory(seed, config, requester)
        try:
            accepted = parse_bool(judge_response)
        except ValueError:
            accepted = False
        print(f"[{attempted}/{len(seeds)}] judge={'kept' if accepted else 'filtered'} id={seed.example_id}")
        if not accepted:
            continue
        kept.append(trajectory_row(seed, messages, judge_response, config))
        progress.update(1)
    progress.close()

    out_path = Path(args.out)
    written = write_jsonl(out_path, kept)
    print(f"Kept {written} / target {args.max_examples} (attempted {attempted} / {len(seeds)}) -> {out_path}")


if __name__ == "__main__":
    main()
