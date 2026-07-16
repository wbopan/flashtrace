"""End-to-end image generation plus multi-token FlashTrace smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from flashtrace import FlashTrace, load_vlm_and_processor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--image", type=Path, default=Path("docs/assets/flashtrace-logo.png")
    )
    parser.add_argument("--prompt", default="What word is written in this image?")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--method", choices=("flashtrace", "ifr-span"), default="flashtrace"
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=56,
        help="Resize the smoke image before Qwen preprocessing to keep the token grid small.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    image = Image.open(args.image).convert("RGB")
    image.thumbnail((args.max_image_side, args.max_image_side))
    model, processor = load_vlm_and_processor(
        args.model,
        dtype="bfloat16",
        device_map="auto",
    )
    tracer = FlashTrace(
        model,
        processor,
        chunk_tokens=64,
        sink_chunk_tokens=8,
        recompute_attention=True,  # VLM path must override this to stored attention.
        generate_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False},
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = tracer.trace(
        prompt=args.prompt,
        images=image,
        method=args.method,
    )
    elapsed = time.perf_counter() - started

    multimodal = result.metadata.get("multimodal", {})
    visual_indices = set(multimodal.get("visual_token_indices_prompt", []))
    visual_scores = [
        {"index": index, "score": float(result.scores[index])}
        for index in sorted(visual_indices)
    ]
    visual_scores.sort(key=lambda item: item["score"], reverse=True)
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "method": args.method,
                "generation": "".join(result.generation_tokens),
                "generation_tokens": len(result.generation_tokens),
                "prompt_features": len(result.prompt_tokens),
                "visual_tokens": len(visual_indices),
                "visual_grid_thw": multimodal.get("visual_grid_thw"),
                "attention_mode": multimodal.get("attention_mode"),
                "top_visual_scores": visual_scores[:5],
                "elapsed_seconds": elapsed,
                "peak_vram_gb": peak_vram_gb,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
