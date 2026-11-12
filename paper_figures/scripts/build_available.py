from __future__ import annotations

import argparse
from pathlib import Path

from plot_efficiency import build as build_efficiency
from plot_mass_shift import build as build_mass_shift
from plot_pilot import build as build_pilot
from plot_reasoning_length import build as build_reasoning_length
from render_heatmaps import DEFAULT_DATA, DEFAULT_MODULE, build as build_heatmaps
from common import DATA_DIR, OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--morehopqa-data-root",
        type=Path,
        default=DATA_DIR / "recovered_proc_1",
        help="Processed NPZ data root recovered from Git commit 075e7e4.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    build_pilot(DATA_DIR / "pilot.json", args.output_dir / "time_plot.pdf")
    print("built time_plot.pdf")
    build_efficiency(DATA_DIR / "efficiency.json", args.output_dir / "cost_comparison.pdf")
    print("built cost_comparison.pdf")
    build_heatmaps(DEFAULT_MODULE, DEFAULT_DATA, args.output_dir)

    build_reasoning_length(args.morehopqa_data_root, args.output_dir / "cot_faithfulness.pdf")
    print("built cot_faithfulness.pdf")
    build_mass_shift(
        args.morehopqa_data_root / "morehopqa" / "qwen-8B" / "ifr_multi_hop_both_n3",
        args.output_dir / "dist_fig.pdf",
    )
    print("built dist_fig.pdf")


if __name__ == "__main__":
    main()
