from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPOS_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PAPER = REPOS_DIR / "icml-flashtrace-tpami"
RECOVERED_ROOT = PACKAGE_DIR / "data" / "recovered_proc_1"
RECOVERED_SOURCE_COMMIT = "075e7e4"
RECOVERED_ARCHIVE_SHA256 = "44666abac156848c095fe141d913afc96370fd090ed0c2994df6dfa912c826ba"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pilot_metrics(path: Path) -> dict[str, float]:
    data = np.load(path, allow_pickle=True)
    vector = np.asarray(data["v_row_all"], dtype=float).reshape(-1)
    prompt_len = int(np.asarray(data["prompt_len"]).item())
    cot_start, cot_end = [int(value) for value in np.asarray(data["thinking_span_gen"]).reshape(-1)]
    out_start, out_end = [int(value) for value in np.asarray(data["sink_span_gen"]).reshape(-1)]
    recovery = float(np.asarray(data["recovery_scores"]).reshape(-1)[1])
    return {
        "input_mass": float(vector[:prompt_len].sum()),
        "cot_mass": float(vector[prompt_len + cot_start : prompt_len + cot_end + 1].sum()),
        "output_mass": float(vector[prompt_len + out_start : prompt_len + out_end + 1].sum()),
        "recovery_at_10pct": recovery,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER)
    args = parser.parse_args()
    math_dir = PACKAGE_DIR / "data" / "math_ifr_multi_hop_both"
    morehop_root = RECOVERED_ROOT / "morehopqa" / "qwen-8B"
    required_counts = {}
    for method in (
        "ifr_multi_hop_both_n1",
        "ifr_multi_hop_both_n3",
        "attnlrp",
        "perturbation_all",
        "perturbation_CLP",
        "perturbation_REAGENT",
    ):
        required_counts[f"morehopqa/{method}"] = len(
            list((morehop_root / method).glob("*.npz"))
        )
    for dataset in ("vt_h2_c3", "vt_h4_c1", "vt_h10_c1.jsonl"):
        for method in (
            "ifr_multi_hop_both_n1",
            "attnlrp",
            "perturbation_all_fast",
            "perturbation_CLP_fast",
            "perturbation_REAGENT_fast",
        ):
            required_counts[f"{dataset}/{method}"] = len(
                list((RECOVERED_ROOT / dataset / "qwen-8B" / method).glob("*.npz"))
            )
    report = {
        "math_heatmap_npz": {
            "path": str(math_dir),
            "count": len(list(math_dir.glob("*.npz"))),
            "complete_expected_100": len(list(math_dir.glob("*.npz"))) == 100,
        },
        "recovered_proc_1": {
            "path": str(RECOVERED_ROOT),
            "source_commit": RECOVERED_SOURCE_COMMIT,
            "source_archive": "exp/proc_1/output/data.zip",
            "source_archive_sha256": RECOVERED_ARCHIVE_SHA256,
            "required_counts": required_counts,
            "complete": all(
                count == (95 if name.startswith("morehopqa/") else 100)
                for name, count in required_counts.items()
            ),
        },
        "published_assets": {},
        "pilot_validation": {},
    }
    pilot = json.loads((PACKAGE_DIR / "data" / "pilot.json").read_text(encoding="utf-8"))
    for sample, directory in zip(pilot["samples"], ("short_cot", "long_cot")):
        recovered = pilot_metrics(
            PACKAGE_DIR / "data" / "pilot_traces" / directory / "ex_000000.npz"
        )
        report["pilot_validation"][sample["label"]] = {
            "recovered": recovered,
            "matches_pilot_json": all(
                np.isclose(recovered[key], float(sample[key]), rtol=0, atol=1e-10)
                for key in recovered
            ),
        }
    for name in (
        "time_plot.pdf",
        "cost_comparison.pdf",
        "cot_faithfulness.pdf",
        "dist_fig.pdf",
        "vis.png",
        "vismath.png",
        "visual_examples.png",
    ):
        path = args.paper_root / "Figures" / name
        report["published_assets"][name] = {
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
