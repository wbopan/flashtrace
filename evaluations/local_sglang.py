"""Strict local-only client for SGLang's native generation endpoint."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlparse


def _loopback_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("SGLang base URL must use http or https.")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Refusing non-loopback SGLang endpoint.")
    return base_url.rstrip("/")


def _finish_reason(meta_info: dict[str, Any]) -> tuple[str, bool]:
    raw = meta_info.get("finish_reason")
    if isinstance(raw, dict):
        reason = str(raw.get("type") or raw.get("reason") or "unknown")
    else:
        reason = str(raw or "unknown")
    normalized = reason.lower()
    stopped = normalized in {"stop", "eos", "end", "end_of_sequence"}
    return reason, stopped


def _output_token_ids(meta_info: dict[str, Any]) -> list[int]:
    rows = meta_info.get("output_token_logprobs") or []
    token_ids = []
    for row in rows:
        token_id = None
        if isinstance(row, dict):
            token_id = row.get("token_id")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            token_id = row[1]
        if token_id is None:
            return []
        token_ids.append(int(token_id))
    return token_ids


@dataclass
class LocalSGLangGenerator:
    base_url: str
    tokenizer: Any
    timeout_seconds: int = 900

    def __post_init__(self) -> None:
        self.base_url = _loopback_base_url(self.base_url)

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SGLang {path} returned HTTP {exc.code}: {body[:4000]}"
            ) from exc

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"base_url": self.base_url}
        for path, key in (
            ("/get_model_info", "model_info"),
            ("/v1/models", "openai_models"),
        ):
            try:
                payload[key] = self._request(path)
            except Exception as exc:
                payload[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        prompt_ids = [
            int(token_id)
            for token_id in self.tokenizer(
                prompt,
                add_special_tokens=False,
            ).input_ids
        ]
        result = self.generate_from_token_ids(
            prompt_ids,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
        )
        result["rendered_prompt"] = prompt
        return result

    def generate_from_token_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Generate from exact token IDs, preserving continuation boundaries."""

        sampling_params: dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k) if top_k > 0 else -1,
            "max_new_tokens": int(max_new_tokens),
            "skip_special_tokens": False,
            "no_stop_trim": True,
        }
        if seed is not None:
            sampling_params["sampling_seed"] = int(seed)
        if stop:
            sampling_params["stop"] = [stop] if isinstance(stop, str) else list(stop)
        request_payload = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "top_logprobs_num": 0,
        }
        started = time.perf_counter()
        response = self._request("/generate", payload=request_payload)
        seconds = time.perf_counter() - started
        if isinstance(response, list):
            if len(response) != 1:
                raise ValueError(
                    f"Expected one SGLang response, received {len(response)}."
                )
            response = response[0]
        if not isinstance(response, dict):
            raise TypeError("SGLang /generate returned a non-object response.")
        raw = str(response.get("text") or "")
        meta_info = dict(response.get("meta_info") or {})
        token_ids = _output_token_ids(meta_info)
        token_id_provenance = "sglang_output_token_logprobs"
        if not token_ids:
            token_ids = [
                int(token_id)
                for token_id in self.tokenizer(
                    raw,
                    add_special_tokens=False,
                ).input_ids
            ]
            token_id_provenance = "retokenized_fallback"
        finish_reason, stopped_on_eos = _finish_reason(meta_info)
        eos_token = self.tokenizer.eos_token
        completion = (
            raw[: -len(eos_token)]
            if eos_token and raw.endswith(eos_token)
            else raw
        )
        return {
            "rendered_prompt": None,
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "generated_token_ids_with_eos_if_emitted": token_ids,
            "generated_token_count_with_eos_if_emitted": len(token_ids),
            "raw_completion_with_eos": raw,
            "completion": completion,
            "stopped_on_eos": stopped_on_eos,
            "finish_reason": finish_reason,
            "generation_seconds": seconds,
            "sampling_params": request_payload["sampling_params"],
            "token_id_provenance": token_id_provenance,
            "sglang_meta_info": meta_info,
        }

    def score_continuation_token_ids(
        self,
        prefix_ids: list[int],
        continuation_ids: list[int],
    ) -> dict[str, Any]:
        """Return aggregate conditional log-probability without generating text."""

        if not prefix_ids or not continuation_ids:
            raise ValueError("Scoring requires non-empty prefix and continuation ids.")
        payload = {
            "input_ids": prefix_ids + continuation_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": len(prefix_ids) - 1,
            "top_logprobs_num": 0,
        }
        started = time.perf_counter()
        response = self._request("/generate", payload=payload)
        seconds = time.perf_counter() - started
        if isinstance(response, list):
            if len(response) != 1:
                raise ValueError("Expected one scoring response.")
            response = response[0]
        rows = list((response.get("meta_info") or {}).get("input_token_logprobs") or [])
        tail = rows[-len(continuation_ids) :]
        logprobs = [
            float(row[0])
            for row in tail
            if isinstance(row, (list, tuple))
            and row
            and isinstance(row[0], (int, float))
        ]
        if len(logprobs) != len(continuation_ids):
            raise ValueError("Scoring response omitted continuation log-probabilities.")
        return {
            "sum_logprob": sum(logprobs),
            "mean_logprob": sum(logprobs) / len(logprobs),
            "scored_token_count": len(logprobs),
            "generation_seconds": seconds,
        }
