from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_REQUIRED_FIELDS = (
    "title",
    "model",
    "method",
    "hops",
    "prompt",
    "generated_text",
    "target_span",
    "reasoning_span",
    "render_model",
    "trace_json",
)

_SUMMARY_FIELDS = ("id", "title", "created_at", "model", "method", "hops", "target_span")
_PREVIEW_CHARS = 160


def _validate_id(sample_id: str) -> str:
    if not sample_id or not _ID_RE.match(sample_id):
        raise ValueError("Invalid sample id.")
    return sample_id


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    summary = {field: record.get(field) for field in _SUMMARY_FIELDS}
    summary["prompt_preview"] = (record.get("prompt") or "")[:_PREVIEW_CHARS]
    return summary


def list_samples(gallery_dir: Path) -> list[dict[str, Any]]:
    directory = Path(gallery_dir)
    if not directory.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append(_summary(record))
    summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return summaries


def load_sample(gallery_dir: Path, sample_id: str) -> dict[str, Any]:
    _validate_id(sample_id)
    path = Path(gallery_dir) / f"{sample_id}.json"
    if not path.is_file():
        raise KeyError(sample_id)
    return json.loads(path.read_text(encoding="utf-8"))


def save_sample(gallery_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in _REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}.")
    directory = Path(gallery_dir)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    sample_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}"
    stored = dict(record)
    stored["id"] = sample_id
    stored["created_at"] = now.isoformat()
    path = directory / f"{sample_id}.json"
    tmp = directory / f".{sample_id}.json.tmp"
    tmp.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return _summary(stored)


def delete_sample(gallery_dir: Path, sample_id: str) -> None:
    _validate_id(sample_id)
    path = Path(gallery_dir) / f"{sample_id}.json"
    if not path.is_file():
        raise KeyError(sample_id)
    path.unlink()
