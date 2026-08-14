from pathlib import Path

import pytest

from evaluations.multimodal.download_strict_artifacts import (
    MANIFEST_PATH,
    _load_manifest,
    _safe_target,
)


def test_strict_artifact_manifest_is_complete_and_unique():
    manifest = _load_manifest(MANIFEST_PATH)
    release = manifest["release"]
    archives = manifest["archives"]

    assert release["tag"] == "multimodal-strict-attribution-v1"
    assert {archive["group"] for archive in archives} == {
        "formal",
        "preview-final",
        "pilot-smoke",
    }

    records = [record for archive in archives for record in archive["files"]]
    paths = [record["path"] for record in records]
    assert len(paths) == 18
    assert len(set(paths)) == len(paths)
    assert all(path.endswith("/attribution_records.jsonl") for path in paths)
    assert all(release["tag"] in archive["url"] for archive in archives)
    assert sum(record["size"] for record in records) == 2_867_867_222


@pytest.mark.parametrize("relative", ["../outside", "/absolute/path"])
def test_strict_artifact_paths_cannot_escape_output_root(tmp_path: Path, relative: str):
    with pytest.raises((RuntimeError, ValueError)):
        _safe_target(tmp_path, relative)
