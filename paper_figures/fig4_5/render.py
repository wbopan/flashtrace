#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib==3.11.1",
#   "numpy==2.5.2",
# ]
# ///
"""Fetch the audited source data and reproduce TPAMI Figures 4 and 5."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from plot_efficiency import build as build_figure_4
from plot_reasoning_length import build as build_figure_5


ROOT = Path(__file__).resolve().parent
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/wbopan/flashtrace/"
    "075e7e44ae4d5acd2ed76e0d2aced57107d02736/exp/proc_1/output/data.zip"
)
ARCHIVE_SHA256 = "44666abac156848c095fe141d913afc96370fd090ed0c2994df6dfa912c826ba"
ARCHIVE_PREFIX = "exp/proc_1/output/data/"
REQUIRED_METHODS = {
    "morehopqa": (
        "ifr_multi_hop_both_n1",
        "attnlrp",
        "perturbation_all",
        "perturbation_CLP",
        "perturbation_REAGENT",
    ),
    "vt_h2_c3": (
        "ifr_multi_hop_both_n1",
        "attnlrp",
        "perturbation_all_fast",
        "perturbation_CLP_fast",
        "perturbation_REAGENT_fast",
    ),
    "vt_h4_c1": (
        "ifr_multi_hop_both_n1",
        "attnlrp",
        "perturbation_all_fast",
        "perturbation_CLP_fast",
        "perturbation_REAGENT_fast",
    ),
    "vt_h10_c1.jsonl": (
        "ifr_multi_hop_both_n1",
        "attnlrp",
        "perturbation_all_fast",
        "perturbation_CLP_fast",
        "perturbation_REAGENT_fast",
    ),
}
EXPECTED_SAMPLES = {
    "morehopqa": 95,
    "vt_h2_c3": 100,
    "vt_h4_c1": 100,
    "vt_h10_c1.jsonl": 100,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(cache_dir: Path) -> Path:
    archive_path = cache_dir / "data.zip"
    if archive_path.is_file() and sha256(archive_path) == ARCHIVE_SHA256:
        print(f"Using verified cached data: {archive_path}")
        return archive_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_dir / "data.zip.part"
    print(f"Downloading recovered Figure 5 data from commit 075e7e4...")
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "flashtrace-tpami-figure-reproducer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary_path.open("wb") as destination:
            shutil.copyfileobj(response, destination)
    observed = sha256(temporary_path)
    if observed != ARCHIVE_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {observed}"
        )
    temporary_path.replace(archive_path)
    print(f"Verified archive SHA-256: {ARCHIVE_SHA256}")
    return archive_path


def selected_member(relative_path: Path) -> bool:
    parts = relative_path.parts
    if len(parts) != 4 or parts[1] != "qwen-8B" or relative_path.suffix != ".npz":
        return False
    dataset, _, method, _ = parts
    return dataset in REQUIRED_METHODS and method in REQUIRED_METHODS[dataset]


def validate_data(data_root: Path) -> bool:
    for dataset, methods in REQUIRED_METHODS.items():
        for method in methods:
            count = len(list((data_root / dataset / "qwen-8B" / method).glob("*.npz")))
            if count != EXPECTED_SAMPLES[dataset]:
                return False
    return True


def extract_figure_5_data(archive_path: Path, cache_dir: Path) -> Path:
    data_root = cache_dir / "data"
    if validate_data(data_root):
        print(f"Using complete extracted Figure 5 data: {data_root}")
        return data_root

    print("Extracting the Figure 5 subsets...")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not member.filename.startswith(ARCHIVE_PREFIX):
                continue
            relative = Path(member.filename.removeprefix(ARCHIVE_PREFIX))
            if not selected_member(relative):
                continue
            destination = data_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    if not validate_data(data_root):
        raise ValueError("The verified archive did not yield the complete Figure 5 subsets.")
    print("Verified Figure 5 sample counts: MoreHopQA 95; each VT subset 100.")
    return data_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    archive_path = fetch_archive(cache_dir)
    figure_5_data = extract_figure_5_data(archive_path, cache_dir)

    figure_4 = output_dir / "cost_comparison.pdf"
    figure_5 = output_dir / "cot_faithfulness.pdf"
    build_figure_4(ROOT / "data" / "efficiency.json", figure_4)
    build_figure_5(figure_5_data, figure_5)

    print(f"TPAMI Fig. 4 written to {figure_4}")
    print(f"TPAMI Fig. 5 written to {figure_5}")
    print("PNG previews were written beside both PDFs.")
    print(
        "Provenance note: Fig. 4 retains the documented legacy interpolation and "
        "placeholder/hand-normalized Pareto values; see README.md."
    )


if __name__ == "__main__":
    main()
