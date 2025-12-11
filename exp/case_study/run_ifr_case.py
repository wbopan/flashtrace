#!/usr/bin/env python3
"""
IFR multi-hop case study runner.

Loads a single sample, runs IFR multi-hop attribution, aggregates hop-wise flow
into sentences, and emits both JSON and HTML reports under exp/case_study/out/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Avoid torchvision dependency when importing transformers (Longformer).
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_TRANSFORMS", "1")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_torchvision() -> None:
    """Provide minimal torchvision stubs so Longformer imports succeed without the real package."""

    if "torchvision" in sys.modules:
        return

    from importlib.machinery import ModuleSpec

    def _mk(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = ModuleSpec(name, loader=None)
        return mod

    tv = _mk("torchvision")
    tv.__dict__["__path__"] = []
    submods = ["transforms", "_meta_registrations", "datasets", "io", "models", "ops", "utils"]
    for name in submods:
        mod = _mk(f"torchvision.{name}")
        sys.modules[f"torchvision.{name}"] = mod
        setattr(tv, name, mod)

    class _InterpolationMode:
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 1
        BICUBIC = 2
        LANCZOS = 3
        BOX = 4
        HAMMING = 5

    sys.modules["torchvision.transforms"].InterpolationMode = _InterpolationMode
    sys.modules["torchvision.transforms"].__all__ = ["InterpolationMode"]

    # ops + misc stub for timm/transformers imports
    ops_mod = sys.modules.get("torchvision.ops") or _mk("torchvision.ops")
    sys.modules["torchvision.ops"] = ops_mod
    setattr(tv, "ops", ops_mod)
    misc_mod = _mk("torchvision.ops.misc")
    sys.modules["torchvision.ops.misc"] = misc_mod
    setattr(ops_mod, "misc", misc_mod)

    class _FrozenBatchNorm2d:
        def __init__(self, *args, **kwargs):
            pass

    misc_mod.FrozenBatchNorm2d = _FrozenBatchNorm2d

    sys.modules["torchvision"] = tv


_stub_torchvision()


def _stub_timm() -> None:
    """Provide minimal timm stubs to avoid optional vision deps."""

    if "timm" in sys.modules:
        return

    from importlib.machinery import ModuleSpec

    def _mk(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = ModuleSpec(name, loader=None)
        return mod

    timm = _mk("timm")
    timm.__dict__["__path__"] = []
    sys.modules["timm"] = timm

    data_mod = _mk("timm.data")
    sys.modules["timm.data"] = data_mod
    timm.data = data_mod

    class _ImageNetInfo:
        pass

    def _infer_imagenet_subset(*args, **kwargs):
        return None

    data_mod.ImageNetInfo = _ImageNetInfo
    data_mod.infer_imagenet_subset = _infer_imagenet_subset

    layers_mod = _mk("timm.layers")
    sys.modules["timm.layers"] = layers_mod
    timm.layers = layers_mod

    create_norm_mod = _mk("timm.layers.create_norm")
    sys.modules["timm.layers.create_norm"] = create_norm_mod
    layers_mod.create_norm = create_norm_mod

    def _get_norm_layer(*args, **kwargs):
        return None

    create_norm_mod.get_norm_layer = _get_norm_layer

    classifier_mod = _mk("timm.layers.classifier")
    sys.modules["timm.layers.classifier"] = classifier_mod
    layers_mod.classifier = classifier_mod


_stub_timm()

import transformers

# Provide light stubs if Longformer classes are unavailable; IFR case study does not use them.
if not hasattr(transformers, "LongformerTokenizer"):
    class _DummyLongformerTokenizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("LongformerTokenizer stubbed; install full transformers+torchvision if needed.")
    transformers.LongformerTokenizer = _DummyLongformerTokenizer

if not hasattr(transformers, "LongformerForMaskedLM"):
    class _DummyLongformerForMaskedLM:
        def __init__(self, *args, **kwargs):
            raise ImportError("LongformerForMaskedLM stubbed; install full transformers+torchvision if needed.")
    transformers.LongformerForMaskedLM = _DummyLongformerForMaskedLM

if hasattr(transformers, "__all__"):
    for _name in ["LongformerTokenizer", "LongformerForMaskedLM"]:
        if _name not in transformers.__all__:
            transformers.__all__.append(_name)

# Gemma3n stubs (transformers may attempt to import even if unused)
if "transformers.models.gemma3n.configuration_gemma3n" not in sys.modules:
    from importlib.machinery import ModuleSpec

    gemma_pkg = types.ModuleType("transformers.models.gemma3n")
    gemma_pkg.__spec__ = ModuleSpec("transformers.models.gemma3n", loader=None, is_package=True)
    sys.modules["transformers.models.gemma3n"] = gemma_pkg

    gemma_conf = types.ModuleType("transformers.models.gemma3n.configuration_gemma3n")
    gemma_conf.__spec__ = ModuleSpec("transformers.models.gemma3n.configuration_gemma3n", loader=None)

    class Gemma3nConfig:
        def __init__(self, *args, **kwargs):
            self.model_type = "gemma3n"

    class Gemma3nTextConfig(Gemma3nConfig):
        pass

    gemma_conf.Gemma3nConfig = Gemma3nConfig
    gemma_conf.Gemma3nTextConfig = Gemma3nTextConfig
    gemma_conf.__all__ = ["Gemma3nConfig", "Gemma3nTextConfig"]
    sys.modules["transformers.models.gemma3n.configuration_gemma3n"] = gemma_conf
    setattr(gemma_pkg, "configuration_gemma3n", gemma_conf)

    if hasattr(transformers, "__all__"):
        for _nm in ["Gemma3nConfig", "Gemma3nTextConfig"]:
            if _nm not in transformers.__all__:
                transformers.__all__.append(_nm)

import llm_attr
from exp.exp2 import dataset_utils as ds_utils
from evaluations.attribution_coverage import load_model

from exp.case_study import analysis, viz


def resolve_device(cuda: Optional[str], cuda_num: int) -> str:
    if cuda and isinstance(cuda, str) and "," in cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        return "auto"
    if cuda and isinstance(cuda, str) and cuda.strip():
        try:
            idx = int(cuda)
        except Exception:
            idx = 0
        return f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    return f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"


def load_example(dataset: str, index: int, data_root: Path) -> Tuple[ds_utils.CachedExample, str]:
    """Load a single example from a cache path or dataset name."""

    ds_path = Path(dataset)
    if ds_path.exists():
        examples = ds_utils.read_cached_jsonl(ds_path)
        dataset_name = ds_path.name
    else:
        loader = ds_utils.DatasetLoader(data_root=data_root)
        examples = loader.load(dataset)
        dataset_name = dataset

    if not examples:
        raise ValueError(f"No examples found for dataset={dataset}")

    if index < 0:
        index = len(examples) + index
    if not (0 <= index < len(examples)):
        raise IndexError(f"index {index} out of range for dataset with {len(examples)} examples")

    return examples[index], dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("IFR multi-hop case study")
    parser.add_argument("--dataset", type=str, default="exp/exp2/data/morehopqa.jsonl", help="Dataset name or JSONL path.")
    parser.add_argument("--data_root", type=str, default="exp/exp2/data", help="Cache root for dataset names.")
    parser.add_argument("--index", type=int, default=0, help="Sample index (supports negative for reverse).")
    parser.add_argument("--model", type=str, default="qwen-8B", help="HF repo id (ignored if --model_path set).")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path to override --model.")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA spec (e.g., '0' or '0,1').")
    parser.add_argument("--cuda_num", type=int, default=0, help="Fallback GPU index when --cuda unset.")
    parser.add_argument("--n_hops", type=int, default=1, help="Number of hops for IFR multi-hop.")
    parser.add_argument("--sink_span", type=int, nargs=2, default=None, help="Optional sink span over generation tokens.")
    parser.add_argument("--thinking_span", type=int, nargs=2, default=None, help="Optional thinking span over generation tokens.")
    parser.add_argument("--chunk_tokens", type=int, default=128, help="IFR chunk size.")
    parser.add_argument("--sink_chunk_tokens", type=int, default=32, help="IFR sink chunk size.")
    parser.add_argument("--output_dir", type=str, default="exp/case_study/out", help="Where to write HTML/JSON artifacts.")
    return parser.parse_args()


def run_ifr(
    example: ds_utils.CachedExample,
    model: Any,
    tokenizer: Any,
    *,
    n_hops: int,
    sink_span: Optional[Sequence[int]],
    thinking_span: Optional[Sequence[int]],
    chunk_tokens: int,
    sink_chunk_tokens: int,
) -> Tuple[Any, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Execute IFR multi-hop attribution for the selected example."""

    attr = llm_attr.LLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=chunk_tokens,
        sink_chunk_tokens=sink_chunk_tokens,
    )

    sink = tuple(sink_span) if sink_span is not None else tuple(example.sink_span) if example.sink_span else None
    thinking = (
        tuple(thinking_span)
        if thinking_span is not None
        else tuple(example.thinking_span) if example.thinking_span else None
    )

    result = attr.calculate_ifr_multi_hop(
        example.prompt,
        target=example.target,
        sink_span=sink,
        thinking_span=thinking,
        n_hops=n_hops,
    )
    return result, sink, thinking


def make_output_stem(dataset_name: str, index: int) -> str:
    safe_name = dataset_name.replace("/", "_").replace(" ", "_")
    return f"ifr_case_{safe_name}_idx{index}"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.cuda, args.cuda_num)

    model_name = args.model_path if args.model_path is not None else args.model
    model, tokenizer = load_model(model_name, device)

    example, ds_name = load_example(args.dataset, args.index, Path(args.data_root))
    attr_result, sink_span, thinking_span = run_ifr(
        example,
        model,
        tokenizer,
        n_hops=args.n_hops,
        sink_span=args.sink_span,
        thinking_span=args.thinking_span,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
    )

    if attr_result.metadata is None or "ifr" not in attr_result.metadata:
        raise RuntimeError("IFR metadata missing from attribution result.")

    ifr_meta = attr_result.metadata["ifr"]
    hop_vectors = ifr_meta.get("per_hop_projected") or []
    if not hop_vectors:
        raise RuntimeError("No per-hop vectors found in IFR metadata.")

    context = analysis.build_sentence_context(
        tokenizer, attr_result.prompt_tokens, attr_result.generation_tokens
    )

    tokens = list(attr_result.prompt_tokens) + list(attr_result.generation_tokens)
    segments = {
        "prompt_len": len(attr_result.prompt_tokens),
        "thinking_span": thinking_span,
        "sink_span": sink_span,
        "generation_len": len(attr_result.generation_tokens),
    }
    hop_records = analysis.package_hops(hop_vectors, context, topk=5)

    case_meta: Dict[str, Any] = {
        "dataset": ds_name,
        "index": args.index,
        "sink_span": sink_span,
        "thinking_span": thinking_span,
        "n_hops": args.n_hops,
        "thinking_ratios": ifr_meta.get("thinking_ratios"),
    }

    context_dict = {
        "prompt_sentences": context.prompt_sentences,
        "generation_sentences": context.generation_sentences,
        "all_sentences": context.all_sentences,
    }

    generation_text = "".join(attr_result.generation_tokens) if attr_result.generation_tokens else ""
    prompt_text = example.prompt
    record = {
        "meta": case_meta,
        "prompt": prompt_text,
        "target": example.target,
        "generation": generation_text,
        "prompt_tokens": attr_result.prompt_tokens,
        "generation_tokens": attr_result.generation_tokens,
        "all_tokens": tokens,
        "segments": segments,
        "prompt_sentences": context.prompt_sentences,
        "generation_sentences": context.generation_sentences,
        "all_sentences": context.all_sentences,
        "hops": hop_records,
        "ifr_meta": analysis.sanitize_ifr_meta(ifr_meta),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = make_output_stem(ds_name, args.index)
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    html = viz.render_case_html(case_meta, context_dict, hop_records, tokens, segments)
    html_path.write_text(html, encoding="utf-8")

    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {html_path}")


if __name__ == "__main__":
    main()
