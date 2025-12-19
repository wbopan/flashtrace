#!/usr/bin/env python3
"""Case study runner for FlashTrace and attribution baselines.

Modes supported (all emit JSON + HTML under ``exp/case_study/out``):

- ``ft``: FlashTrace (current project implementation; multi-hop IFR)
- ``ifr``: standard IFR single-hop visualization
- ``attnlrp``: AttnLRP hop0 (reuse FT-AttnLRP span-aggregate; visualize raw hop0 vector)
- ``ft_attnlrp``: FT-AttnLRP (multi-hop aggregated AttnLRP; matches exp/exp2)
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

def _early_set_cuda_visible_devices() -> None:
    """Set CUDA_VISIBLE_DEVICES before importing torch/transformers.

    Note: CUDA device indices are re-mapped inside the process after applying the mask.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    cuda = args.cuda.strip() if isinstance(args.cuda, str) else ""
    if cuda and "," in cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda


if __name__ == "__main__":
    _early_set_cuda_visible_devices()

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
from evaluations.attribution_recovery import load_model

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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ft", "ifr", "attnlrp", "ft_attnlrp"],
        default="ft",
        help="ft = FlashTrace (multi-hop IFR); ifr = standard IFR; attnlrp = AttnLRP hop0 (FT-AttnLRP span-aggregate); ft_attnlrp = FT-AttnLRP (multi-hop aggregated; exp2).",
    )
    parser.add_argument(
        "--ifr_view",
        type=str,
        choices=["aggregate", "per_token"],
        default="aggregate",
        help="Only for --mode ifr. aggregate = sink-span aggregated IFR (one panel); per_token = IFR per sink token (one panel per token).",
    )
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


def run_ft_multihop(
    example: ds_utils.CachedExample,
    model: Any,
    tokenizer: Any,
    *,
    n_hops: int,
    sink_span: Optional[Sequence[int]],
    thinking_span: Optional[Sequence[int]],
    chunk_tokens: int,
    sink_chunk_tokens: int,
) -> Tuple[Any, Optional[Tuple[int, int]], Optional[Tuple[int, int]], Dict[str, Any]]:
    """Execute FT (current multi-hop IFR) attribution for the selected example."""

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
    debug_info: Dict[str, Any] = {
        "full_prompt_tokens": list(getattr(attr, "prompt_tokens", []) or []),
        "generation_tokens": list(getattr(attr, "generation_tokens", []) or []),
        "user_prompt_indices": list(getattr(attr, "user_prompt_indices", []) or []),
        "chat_prompt_indices": list(getattr(attr, "chat_prompt_indices", []) or []),
        "prompt_ids": getattr(attr, "prompt_ids", None).detach().cpu().tolist() if getattr(attr, "prompt_ids", None) is not None else None,
        "generation_ids": getattr(attr, "generation_ids", None).detach().cpu().tolist() if getattr(attr, "generation_ids", None) is not None else None,
    }

    raw_vectors = []
    if result.metadata and "ifr" in result.metadata:
        raw_ifr = result.metadata["ifr"].get("raw")
        if raw_ifr is not None and hasattr(raw_ifr, "raw_attributions"):
            try:
                raw_vectors = [r.token_importance_total.detach().cpu() for r in raw_ifr.raw_attributions]
            except Exception:
                raw_vectors = []
    debug_info["raw_hop_vectors"] = raw_vectors

    return result, sink, thinking, debug_info


def make_output_stem(dataset_name: str, index: int, mode: str) -> str:
    safe_name = dataset_name.replace("/", "_").replace(" ", "_")
    prefix = {
        "ft": "ft_case_",
        "ifr": "ifr_case_",
        "attnlrp": "attnlrp_case_",
        "ft_attnlrp": "ft_attnlrp_case_",
    }.get(mode, f"{mode}_case_")
    return f"{prefix}{safe_name}_idx{index}"


def _decode_token_ids(tokenizer: Any, ids: Sequence[int]) -> List[str]:
    """Decode each token id into a readable text piece (keeps special tokens)."""

    pieces: List[str] = []
    for tok_id in ids:
        try:
            pieces.append(
                tokenizer.decode([int(tok_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
        except Exception:
            pieces.append(str(tok_id))
    return pieces


def build_raw_tokens_from_ids(tokenizer: Any, prompt_ids: Optional[Sequence[int]], generation_ids: Optional[Sequence[int]]) -> List[str]:
    if not prompt_ids:
        prompt_ids = []
    if not generation_ids:
        generation_ids = []
    return _decode_token_ids(tokenizer, prompt_ids) + _decode_token_ids(tokenizer, generation_ids)


def build_trimmed_roles(tokens: Sequence[str], segments: Dict[str, Any]) -> List[str]:
    """Assign role labels for trimmed tokens (prompt + generation)."""

    roles = ["prompt" for _ in range(len(tokens))]
    prompt_len_tokens = segments.get("prompt_len", 0)
    for idx in range(prompt_len_tokens, len(tokens)):
        roles[idx] = "gen"
    thinking_span = segments.get("thinking_span")
    sink_span = segments.get("sink_span")
    if thinking_span is not None:
        start = prompt_len_tokens + int(thinking_span[0])
        end = prompt_len_tokens + int(thinking_span[1])
        for i in range(start, min(len(tokens), end + 1)):
            roles[i] = "think"
    if sink_span is not None:
        start = prompt_len_tokens + int(sink_span[0])
        end = prompt_len_tokens + int(sink_span[1])
        for i in range(start, min(len(tokens), end + 1)):
            roles[i] = "output"
    return roles


def build_raw_roles(
    tokens: Sequence[str],
    prompt_len_full: int,
    user_indices: Sequence[int],
    template_indices: Sequence[int],
    thinking_span_abs: Optional[Sequence[int]],
    sink_span_abs: Optional[Sequence[int]],
) -> List[str]:
    """Assign role labels for raw tokens (template + user + generation)."""

    roles = ["template" for _ in range(len(tokens))]
    user_set = set(int(i) for i in user_indices)
    tmpl_set = set(int(i) for i in template_indices)

    for i in range(min(len(tokens), prompt_len_full)):
        if i in user_set:
            roles[i] = "user"
        elif i in tmpl_set:
            roles[i] = "template"
        else:
            roles[i] = "prompt"

    for i in range(prompt_len_full, len(tokens)):
        roles[i] = "gen"

    if thinking_span_abs is not None:
        start, end = int(thinking_span_abs[0]), int(thinking_span_abs[1])
        for i in range(start, min(len(tokens), end + 1)):
            roles[i] = "think"

    if sink_span_abs is not None:
        start, end = int(sink_span_abs[0]), int(sink_span_abs[1])
        for i in range(start, min(len(tokens), end + 1)):
            roles[i] = "output"

    return roles


def main() -> None:
    args = parse_args()
    device = resolve_device(args.cuda, args.cuda_num)
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(f"[info] CUDA_VISIBLE_DEVICES={visible!r} torch.cuda.device_count()={torch.cuda.device_count()} device={device}")

    model_name = args.model_path if args.model_path is not None else args.model
    # Align with exp/exp2: always use the shared fp16 loader.
    model, tokenizer = load_model(model_name, device)

    example, ds_name = load_example(args.dataset, args.index, Path(args.data_root))
    mode = args.mode

    sink_span: Optional[Tuple[int, int]] = None
    thinking_span: Optional[Tuple[int, int]] = None
    thinking_ratios: Optional[Sequence[float]] = None

    prompt_tokens_trimmed: List[str] = []
    generation_tokens_trimmed: List[str] = []
    hop_vectors_trimmed: List[torch.Tensor] = []
    hop_vectors_raw: List[torch.Tensor] = []
    prompt_len_full: Optional[int] = None
    user_prompt_indices: List[int] = []
    chat_prompt_indices: List[int] = []
    method_meta: Dict[str, Any] = {}
    raw_prompt_ids: Optional[List[int]] = None
    raw_generation_ids: Optional[List[int]] = None

    if mode == "ft":
        attr_result, sink_span, thinking_span, debug_info = run_ft_multihop(
            example,
            model,
            tokenizer,
            n_hops=args.n_hops,
            sink_span=args.sink_span,
            thinking_span=args.thinking_span,
            chunk_tokens=args.chunk_tokens,
            sink_chunk_tokens=args.sink_chunk_tokens,
        )
        ifr_meta = (attr_result.metadata or {}).get("ifr") or {}
        hop_vectors_trimmed = list(ifr_meta.get("per_hop_projected") or [])
        if not hop_vectors_trimmed:
            raise RuntimeError("No per-hop vectors found for ft mode.")

        prompt_tokens_trimmed = list(attr_result.prompt_tokens)
        generation_tokens_trimmed = list(attr_result.generation_tokens)
        thinking_ratios = ifr_meta.get("thinking_ratios")

        raw_prompt_ids = debug_info.get("prompt_ids")
        if isinstance(raw_prompt_ids, list) and raw_prompt_ids and isinstance(raw_prompt_ids[0], list):
            raw_prompt_ids = raw_prompt_ids[0]
        raw_generation_ids = debug_info.get("generation_ids")
        if isinstance(raw_generation_ids, list) and raw_generation_ids and isinstance(raw_generation_ids[0], list):
            raw_generation_ids = raw_generation_ids[0]

        user_prompt_indices = list(debug_info.get("user_prompt_indices") or [])
        chat_prompt_indices = list(debug_info.get("chat_prompt_indices") or [])
        prompt_len_full = len(raw_prompt_ids) if isinstance(raw_prompt_ids, list) else None

        raw_vectors = debug_info.get("raw_hop_vectors") or []
        hop_vectors_raw = [vec.detach().cpu() if hasattr(vec, "detach") else torch.as_tensor(vec) for vec in raw_vectors]
        method_meta = {"ifr": analysis.sanitize_ifr_meta(ifr_meta)}

    elif mode == "ifr":
        # Standard IFR (single-hop), with pre/post trim views.
        attr = llm_attr.LLMIFRAttribution(
            model,
            tokenizer,
            chunk_tokens=args.chunk_tokens,
            sink_chunk_tokens=args.sink_chunk_tokens,
        )
        sink_span = tuple(args.sink_span) if args.sink_span is not None else tuple(example.sink_span) if example.sink_span else None
        thinking_span = tuple(args.thinking_span) if args.thinking_span is not None else tuple(example.thinking_span) if example.thinking_span else sink_span

        if sink_span is None:
            raise ValueError("sink_span is required for IFR mode (use dataset sink_span or pass --sink_span).")

        if args.ifr_view == "aggregate":
            span_result = attr.calculate_ifr_span(
                example.prompt,
                target=example.target,
                span=tuple(sink_span),
            )
            span_meta = span_result.metadata.get("ifr") if span_result.metadata else None
            aggregate = span_meta.get("aggregate") if isinstance(span_meta, dict) else None
            if aggregate is None or not hasattr(aggregate, "token_importance_total"):
                raise RuntimeError("IFR span aggregate missing from metadata; cannot render pre-trim view.")

            raw_vector = aggregate.token_importance_total.detach().cpu()
            trimmed_vector = attr._project_vector(raw_vector)
            hop_vectors_raw = [raw_vector]
            hop_vectors_trimmed = [trimmed_vector]

            prompt_tokens_trimmed = list(attr.user_prompt_tokens)
            generation_tokens_trimmed = list(attr.generation_tokens)

            raw_prompt_ids = attr.prompt_ids.detach().cpu().tolist()[0]
            raw_generation_ids = attr.generation_ids.detach().cpu().tolist()[0]
            user_prompt_indices = list(getattr(attr, "user_prompt_indices", []) or [])
            chat_prompt_indices = list(getattr(attr, "chat_prompt_indices", []) or [])
            prompt_len_full = len(raw_prompt_ids)

            sink_abs = (prompt_len_full + sink_span[0], prompt_len_full + sink_span[1])
            think_abs = (prompt_len_full + thinking_span[0], prompt_len_full + thinking_span[1]) if thinking_span else None

            meta = {
                "type": "span_aggregate",
                "ifr_view": "aggregate",
                "sink_span_generation": sink_span,
                "sink_span_absolute": sink_abs,
                "thinking_span_generation": thinking_span,
                "thinking_span_absolute": think_abs,
            }
            method_meta = {"ifr": analysis.tensor_to_list(meta)}
        else:
            input_ids_all, attn_mask, prompt_len_full_tmp, gen_len = attr._ensure_generation(example.prompt, example.target)
            total_len = int(input_ids_all.shape[1])

            cache, attentions, metadata_full, weight_pack = attr._capture_model_state(input_ids_all, attn_mask)
            params = attr._build_ifr_params(metadata_full, total_len)
            renorm = attr.renorm_threshold_default

            sink_range = (prompt_len_full_tmp, prompt_len_full_tmp + gen_len - 1)
            all_positions = llm_attr.compute_ifr_for_all_positions(
                cache=cache,
                attentions=attentions,
                weight_pack=weight_pack,
                params=params,
                renorm_threshold=renorm,
                sink_range=sink_range,
                return_layerwise=False,
            )

            span_start, span_end = sink_span
            if span_start < 0 or span_end >= gen_len:
                raise ValueError(f"Invalid sink_span {sink_span} for generation length {gen_len}")
            sink_abs_start = prompt_len_full_tmp + span_start
            sink_abs_end = prompt_len_full_tmp + span_end

            raw_vectors = []
            if all_positions.token_importance_matrix.numel() > 0:
                mat = all_positions.token_importance_matrix.detach().cpu()
                raw_vectors = [mat[idx] for idx in range(span_start, span_end + 1)]

            trimmed_vectors = []
            for vec in raw_vectors:
                projected = attr.extract_user_prompt_attributions(attr.prompt_tokens, vec.view(1, -1))[0]
                trimmed_vectors.append(projected)

            hop_vectors_raw = [vec.detach().cpu() if hasattr(vec, "detach") else torch.as_tensor(vec) for vec in raw_vectors]
            hop_vectors_trimmed = trimmed_vectors

            prompt_tokens_trimmed = list(attr.user_prompt_tokens)
            generation_tokens_trimmed = list(attr.generation_tokens)

            raw_prompt_ids = attr.prompt_ids.detach().cpu().tolist()[0]
            raw_generation_ids = attr.generation_ids.detach().cpu().tolist()[0]
            user_prompt_indices = list(getattr(attr, "user_prompt_indices", []) or [])
            chat_prompt_indices = list(getattr(attr, "chat_prompt_indices", []) or [])
            prompt_len_full = len(raw_prompt_ids)

            meta = {
                "type": "all_positions_subset",
                "ifr_view": "per_token",
                "sink_span_generation": sink_span,
                "sink_span_absolute": (sink_abs_start, sink_abs_end),
                "thinking_span_generation": thinking_span,
                "thinking_span_absolute": (prompt_len_full + thinking_span[0], prompt_len_full + thinking_span[1]) if thinking_span else None,
            }
            method_meta = {"ifr": analysis.tensor_to_list(meta)}

    elif mode in ("attnlrp", "ft_attnlrp"):
        # Reuse the shared LLMLRPAttribution implementations (root-level).
        attributor = llm_attr.LLMLRPAttribution(model, tokenizer)

        sink_span = tuple(args.sink_span) if args.sink_span is not None else tuple(example.sink_span) if example.sink_span else None
        thinking_span = (
            tuple(args.thinking_span)
            if args.thinking_span is not None
            else tuple(example.thinking_span) if example.thinking_span else sink_span
        )

        if mode == "attnlrp":
            # Case-study AttnLRP: reuse FT-AttnLRP logic but take hop0 (the first span-aggregate)
            # for a full, signed attribution vector (no observation masking).
            multi_hop = attributor.calculate_attnlrp_multi_hop(
                example.prompt,
                target=example.target,
                sink_span=sink_span,
                thinking_span=thinking_span,
                n_hops=0,
            )
            base_attr = (getattr(multi_hop, "raw_attributions", None) or [None])[0]
            if base_attr is None or not hasattr(base_attr, "token_importance_total"):
                raise RuntimeError("AttnLRP hop0 missing from multi-hop result.")

            hop0_vec = torch.as_tensor(getattr(base_attr, "token_importance_total"), dtype=torch.float32).detach().cpu()
            if hop0_vec.numel() <= 0:
                raise RuntimeError("Empty generation for AttnLRP case study.")

            # Use the actual sink span applied by hop0 (defaults to full generation when unset).
            sink_span = tuple(getattr(base_attr, "sink_range"))
            if thinking_span is None:
                thinking_span = sink_span

            hop_vectors_trimmed = [hop0_vec]
            thinking_ratios = list(getattr(multi_hop, "thinking_ratios", []) or [])

            method_meta = {
                "attnlrp": {
                    "type": "calculate_attnlrp_multi_hop(n_hops=0) hop0 raw_attributions[0]",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "thinking_ratios": thinking_ratios,
                }
            }
        else:
            # exp2 ft_attnlrp: multi-hop aggregated AttnLRP (metadata contains per-hop vectors).
            attr_result = attributor.calculate_attnlrp_aggregated_multi_hop(
                example.prompt,
                target=example.target,
                sink_span=sink_span,
                thinking_span=thinking_span,
                n_hops=int(args.n_hops),
            )
            meta = attr_result.metadata or {}
            multi_hop = meta.get("multi_hop_result")
            if multi_hop is None:
                raise RuntimeError("FT-AttnLRP case study missing metadata.multi_hop_result.")

            raw_attributions = getattr(multi_hop, "raw_attributions", None) or []
            hop_vectors_trimmed = [
                torch.as_tensor(getattr(hop, "token_importance_total"), dtype=torch.float32).detach().cpu()
                for hop in raw_attributions
            ]
            thinking_ratios = list(getattr(multi_hop, "thinking_ratios", []) or [])

            method_meta = {
                "attnlrp": {
                    "type": "calculate_attnlrp_aggregated_multi_hop (exp2 ft_attnlrp)",
                    "n_hops": int(args.n_hops),
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "thinking_ratios": thinking_ratios,
                }
            }

        prompt_tokens_trimmed = list(attributor.user_prompt_tokens)
        generation_tokens_trimmed = list(attributor.generation_tokens)

        raw_prompt_ids = attributor.prompt_ids.detach().cpu().tolist()[0]
        raw_generation_ids = attributor.generation_ids.detach().cpu().tolist()[0]
        user_prompt_indices = list(getattr(attributor, "user_prompt_indices", []) or [])
        chat_prompt_indices = list(getattr(attributor, "chat_prompt_indices", []) or [])
        prompt_len_full = len(raw_prompt_ids)

    else:
        raise ValueError(f"Unsupported mode={mode}")

    if not hop_vectors_trimmed:
        raise RuntimeError("No hop vectors to visualize.")

    tokens = list(prompt_tokens_trimmed) + list(generation_tokens_trimmed)
    raw_tokens = build_raw_tokens_from_ids(tokenizer, raw_prompt_ids, raw_generation_ids)

    segments = {
        "prompt_len": len(prompt_tokens_trimmed),
        "thinking_span": thinking_span,
        "sink_span": sink_span,
        "generation_len": len(generation_tokens_trimmed),
    }
    roles_trimmed = build_trimmed_roles(tokens, segments)

    sink_span_abs = None
    thinking_span_abs = None
    if prompt_len_full is not None and sink_span is not None:
        sink_span_abs = (prompt_len_full + sink_span[0], prompt_len_full + sink_span[1])
    if prompt_len_full is not None and thinking_span is not None:
        thinking_span_abs = (prompt_len_full + thinking_span[0], prompt_len_full + thinking_span[1])
    prompt_len_full_safe = int(prompt_len_full or 0)
    roles_raw = build_raw_roles(
        raw_tokens,
        prompt_len_full_safe,
        user_prompt_indices,
        chat_prompt_indices,
        thinking_span_abs,
        sink_span_abs,
    )

    # Visualize signed scores: blue = negative, red = positive.
    score_transform = "signed"

    # Lightweight debug stats to catch silent all-zero / NaN cases.
    hop_stats_raw = [analysis.vector_stats(torch.nan_to_num(v.detach().cpu(), nan=0.0)) for v in hop_vectors_raw]
    hop_stats_trimmed = [analysis.vector_stats(torch.nan_to_num(v.detach().cpu(), nan=0.0)) for v in hop_vectors_trimmed]
    for i in range(max(len(hop_stats_raw), len(hop_stats_trimmed))):
        raw_abs = hop_stats_raw[i]["abs_max"] if i < len(hop_stats_raw) else None
        trim_abs = hop_stats_trimmed[i]["abs_max"] if i < len(hop_stats_trimmed) else None
        print(f"[stats] panel {i}: raw_abs_max={raw_abs} trimmed_abs_max={trim_abs}")

    hop_token_trim = analysis.package_token_hops(hop_vectors_trimmed, transform=score_transform)
    hop_token_raw = analysis.package_token_hops(hop_vectors_raw, transform=score_transform) if hop_vectors_raw else []

    case_meta: Dict[str, Any] = {
        "dataset": ds_name,
        "index": args.index,
        "sink_span": sink_span,
        "thinking_span": thinking_span,
        "n_hops": args.n_hops,
        "thinking_ratios": thinking_ratios,
        "mode": mode,
        "ifr_view": method_meta.get("ifr", {}).get("ifr_view") if isinstance(method_meta.get("ifr"), dict) else None,
        "score_transform": score_transform,
        "vector_stats_raw": hop_stats_raw,
        "vector_stats_trimmed": hop_stats_trimmed,
    }

    generation_text = "".join(generation_tokens_trimmed) if generation_tokens_trimmed else ""
    prompt_text = example.prompt
    record = {
        "meta": case_meta,
        "prompt": prompt_text,
        "target": example.target,
        "generation": generation_text,
        "prompt_tokens": prompt_tokens_trimmed,
        "generation_tokens": generation_tokens_trimmed,
        "all_tokens": tokens,
        "full_prompt_tokens": raw_tokens[:prompt_len_full_safe] if prompt_len_full_safe else [],
        "full_all_tokens": raw_tokens,
        "token_roles": roles_trimmed,
        "raw_token_roles": roles_raw,
        "segments": segments,
        "token_hops_trimmed": hop_token_trim,
        "token_hops_raw": hop_token_raw,
        "ifr_meta": method_meta.get("ifr"),
        "attnlrp_meta": method_meta.get("attnlrp"),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = make_output_stem(ds_name, args.index, mode)
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    html = viz.render_case_html(
        case_meta,
        token_view_trimmed={
            "label": "Post-trim token-level heatmap (user prompt + generation; chat template removed)",
            "tokens": tokens,
            "roles": roles_trimmed,
            "hops": hop_token_trim,
        },
        token_view_raw=(
            {
                "label": "Pre-trim token-level heatmap (with chat template)",
                "tokens": raw_tokens,
                "roles": roles_raw,
                "hops": hop_token_raw,
            }
            if hop_token_raw
            else None
        ),
    )
    html_path.write_text(html, encoding="utf-8")

    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {html_path}")


if __name__ == "__main__":
    main()
