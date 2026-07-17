"""Download VQA-X/A-OKVQA annotations and selected COCO images.

The original VQA-X Google Drive folder linked by Park et al. currently returns
404.  We therefore use the structured VQA-X release from the authors of
NLX-GPT, which preserves questions, ten VQA answers, human explanations, COCO
image IDs, and the published train/validation/test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Iterable

from .datasets import MultimodalExample, load_examples


AOKVQA_URL = (
    "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/"
    "aokvqa_v1p0.tar.gz"
)
AOKVQA_SHA256 = "3992b488babc0c1147f0def18c7a55274aeeb37ab668cf80226b8f62ee35a8e1"
VQAX_FILES = {
    "train": (
        "1GOncMUfGvwUfmcT3rR02qefezwI9jBSa",
        "ab16ca319b86179be98ea7de34ec14f8d065508d0507adbce30150a78c5ba114",
    ),
    "val": (
        "1CyA_seP2RUXrKmtb2BuCPUJ71D91AqcH",
        "18e5ae0f42c8bf1b52cadd11a125eab765783a94f86fdb0ae7bfba9d297dbfeb",
    ),
    "test": (
        "1LD2hW6Ul4xzz5dFTofW3s5H4IinltJel",
        "7ae5f0e930276040b8cdcc0d1d63b2ee18564a908d67025de48dc08f4a99f264",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, sha256: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (sha256 is None or _sha256(destination) == sha256):
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FlashTrace-eval/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output)
            if sha256 is not None and _sha256(temporary) != sha256:
                raise RuntimeError(f"Checksum mismatch for {url}")
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        try:
            tar.extractall(destination, filter="data")
        except TypeError:  # Python 3.10/3.11, after the path validation above.
            tar.extractall(destination)


def prepare_annotations(data_root: Path) -> dict[str, object]:
    aok_dir = data_root / "aokvqa"
    archive = aok_dir / "aokvqa_v1p0.tar.gz"
    _download(AOKVQA_URL, archive, sha256=AOKVQA_SHA256)
    if not (aok_dir / "aokvqa_v1p0_val.json").exists():
        _safe_extract(archive, aok_dir)

    vqax_dir = data_root / "vqa_x" / "nlxgpt"
    sources: dict[str, str] = {}
    for split, (file_id, checksum) in VQAX_FILES.items():
        url = (
            "https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm=t"
        )
        _download(url, vqax_dir / f"vqaX_{split}.json", sha256=checksum)
        sources[split] = url
    return {
        "aokvqa": {"url": AOKVQA_URL, "sha256": AOKVQA_SHA256},
        "vqa_x": {
            "source": "NLX-GPT structured VQA-X release",
            "urls": sources,
            "sha256": {split: value[1] for split, value in VQAX_FILES.items()},
        },
    }


def _image_url(example: MultimodalExample) -> str:
    # COCO's own download instructions use HTTP. Its HTTPS endpoint currently
    # serves a certificate without images.cocodataset.org in the SAN list.
    return f"http://images.cocodataset.org/{example.coco_split}/{example.image_path.name}"


def prepare_images(examples: Iterable[MultimodalExample]) -> list[dict[str, object]]:
    records = []
    for example in examples:
        url = _image_url(example)
        _download(url, example.image_path)
        records.append(
            {
                "dataset": example.dataset,
                "question_id": example.question_id,
                "image": str(example.image_path),
                "url": url,
                "sha256": _sha256(example.image_path),
            }
        )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--dataset", choices=("all", "vqa_x", "aokvqa"), default="all"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    annotation_sources = prepare_annotations(args.data_root)
    datasets = ("vqa_x", "aokvqa") if args.dataset == "all" else (args.dataset,)
    examples = [
        example
        for dataset in datasets
        for example in load_examples(
            dataset, args.data_root, split=args.split, limit=args.samples
        )
    ]
    images = prepare_images(examples)
    manifest = {
        "split": args.split,
        "samples_per_dataset": args.samples,
        "annotation_sources": annotation_sources,
        "images": images,
    }
    manifest_path = args.data_root / "multimodal_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "images": len(images)}, indent=2))


if __name__ == "__main__":
    main()
