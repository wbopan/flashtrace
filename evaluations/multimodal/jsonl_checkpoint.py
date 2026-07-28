"""Crash-safe incremental checkpoints for large paired JSONL evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl, write_jsonl


class PairJsonlCheckpoint:
    """Journal pair records without rewriting the full JSONL after every item.

    The canonical JSONL remains unchanged while work is in progress. Each
    sample/method result is durably written to a small atomic shard. A resumed
    process overlays those shards on the last canonical snapshot, and
    :meth:`compact` produces the same single-file artifact expected by
    downstream analysis.
    """

    def __init__(
        self,
        snapshot_path: Path,
        *,
        key_fields: Sequence[str] = ("sample_id", "method"),
    ) -> None:
        self.snapshot_path = snapshot_path
        self.key_fields = tuple(key_fields)
        if not self.key_fields:
            raise ValueError("checkpoint key_fields must be non-empty")
        self.journal_dir = snapshot_path.with_name(f".{snapshot_path.name}.journal")
        self._records: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()

        if snapshot_path.exists():
            for record in read_jsonl(snapshot_path):
                self._replace_in_memory(record)
        if self.journal_dir.exists():
            for shard in sorted(self.journal_dir.glob("*.json")):
                with shard.open(encoding="utf-8") as handle:
                    self._replace_in_memory(json.load(handle))

    def _key(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        missing = [field for field in self.key_fields if field not in record]
        if missing:
            raise ValueError(f"checkpoint record is missing key fields: {missing}")
        return tuple(str(record[field]) for field in self.key_fields)

    def _replace_in_memory(self, record: Mapping[str, Any]) -> None:
        materialized = dict(record)
        key = self._key(materialized)
        self._records.pop(key, None)
        self._records[key] = materialized

    def _shard_path(self, key: tuple[str, ...]) -> Path:
        encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return self.journal_dir / f"{digest}.json"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def put(self, record: Mapping[str, Any]) -> None:
        """Atomically checkpoint one pair, replacing an earlier attempt."""

        materialized = dict(record)
        key = self._key(materialized)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        destination = self._shard_path(key)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.journal_dir,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(materialized, handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
            self._fsync_directory(self.journal_dir)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        self._replace_in_memory(materialized)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records.values())

    def compact(self) -> int:
        """Write one canonical snapshot, then retire incorporated shards."""

        count = write_jsonl(self.records(), self.snapshot_path)
        if self.journal_dir.exists():
            for shard in self.journal_dir.glob("*.json"):
                shard.unlink()
            for temporary in self.journal_dir.glob(".*.tmp"):
                temporary.unlink()
            try:
                self.journal_dir.rmdir()
            except OSError:
                pass
        return count
