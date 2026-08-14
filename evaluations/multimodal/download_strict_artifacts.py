"""Restore checksum-pinned strict attribution records from GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any
from urllib.request import Request, urlopen


MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "strict-attribution-records-v1.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, record: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == record["size"]
        and _sha256(path) == record["sha256"]
    )


def _download(archive: dict[str, Any], cache_dir: Path) -> Path:
    destination = cache_dir / archive["asset"]
    if _matches(destination, archive):
        return destination

    cache_dir.mkdir(parents=True, exist_ok=True)
    request = Request(
        archive["url"],
        headers={"User-Agent": "flashtrace-artifact-downloader/1"},
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive['asset']}.", suffix=".part", dir=cache_dir, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(request) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    if not _matches(temporary_path, archive):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"archive checksum or size mismatch: {archive['asset']}")
    os.replace(temporary_path, destination)
    return destination


def _safe_target(output_root: Path, relative: str) -> Path:
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise RuntimeError(f"unsafe manifest path: {relative}")
    target = output_root.joinpath(*posix_path.parts)
    target.resolve().relative_to(output_root.resolve())
    return target


def _extract(
    archive_path: Path,
    archive: dict[str, Any],
    output_root: Path,
    force: bool,
) -> tuple[int, int]:
    expected = {record["path"]: record for record in archive["files"]}
    if not force:
        for relative, record in expected.items():
            target = _safe_target(output_root, relative)
            if target.exists() and not _matches(target, record):
                raise RuntimeError(
                    f"existing file does not match manifest: {target}; "
                    "rerun with --force to replace it"
                )
    restored = 0
    kept = 0
    with tarfile.open(archive_path, mode="r:gz") as bundle:
        members = bundle.getmembers()
        actual = {member.name for member in members}
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise RuntimeError(
                f"archive member mismatch for {archive['asset']}: "
                f"missing={missing}, extra={extra}"
            )

        for member in members:
            if not member.isfile():
                raise RuntimeError(f"non-file archive member rejected: {member.name}")
            record = expected[member.name]
            target = _safe_target(output_root, member.name)
            if target.exists():
                if _matches(target, record):
                    kept += 1
                    continue

            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    shutil.copyfileobj(source, temporary, length=1024 * 1024)
                except BaseException:
                    temporary_path.unlink(missing_ok=True)
                    raise
            if not _matches(temporary_path, record):
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(f"restored file checksum mismatch: {member.name}")
            os.replace(temporary_path, target)
            restored += 1
    return restored, kept


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"unsupported manifest schema: {manifest.get('schema_version')}")
    return manifest


def _default_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "flashtrace" / "release-assets"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        choices=("formal", "preview-final", "pilot-smoke"),
        help="restore only this group; repeat for multiple groups (default: all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root receiving restored relative paths",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="download cache for verified release archives",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing files whose checksum does not match the manifest",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available groups and files without downloading",
    )
    args = parser.parse_args()

    manifest = _load_manifest(MANIFEST_PATH)
    selected_groups = set(args.group or [])
    archives = [
        archive
        for archive in manifest["archives"]
        if not selected_groups or archive["group"] in selected_groups
    ]

    if args.list:
        for archive in archives:
            print(f"{archive['group']}: {archive['asset']} ({archive['size']} bytes)")
            for record in archive["files"]:
                print(f"  {record['path']} ({record['size']} bytes)")
        return 0

    restored_total = 0
    kept_total = 0
    output_root = args.output_root.resolve()
    for archive in archives:
        records = archive["files"]
        already_complete = all(
            _matches(_safe_target(output_root, record["path"]), record)
            for record in records
        )
        if already_complete:
            kept_total += len(records)
            print(f"verified {archive['group']}: all files already present")
            continue

        print(f"downloading {archive['url']}")
        archive_path = _download(archive, args.cache_dir.expanduser())
        restored, kept = _extract(archive_path, archive, output_root, args.force)
        restored_total += restored
        kept_total += kept
        print(f"verified {archive['group']}: restored={restored}, existing={kept}")

    print(f"complete: restored={restored_total}, existing={kept_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
