from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


DEFAULT_MODULE = Path(__file__).resolve().parent / "legacy_visualize_heatmap.py"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "math_ifr_multi_hop_both"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("flashtrace_legacy_heatmap", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import heatmap module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(module_path: Path, data_dir: Path, output_dir: Path) -> None:
    module = load_module(module_path)
    expected = [data_dir / f"ex_{index:06d}.npz" for index in (0, 17, 40, 54)]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing heatmap NPZ files: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_html = module.generate_multi_sample_html([(str(data_dir / "ex_000040.npz"), "")])
    appendix_html = module.generate_multi_sample_html(
        [
            (str(data_dir / "ex_000017.npz"), "Math #17"),
            (str(data_dir / "ex_000054.npz"), "Math #54"),
            (str(data_dir / "ex_000000.npz"), "Math #00"),
        ]
    )
    (output_dir / "vis.html").write_text(main_html, encoding="utf-8")
    (output_dir / "vismath.html").write_text(appendix_html, encoding="utf-8")
    print(f"Wrote {output_dir / 'vis.html'}")
    print(f"Wrote {output_dir / 'vismath.html'}")
    print("The published PNGs were browser screenshots; viewport and scale are not recorded in the repository.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.module, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
