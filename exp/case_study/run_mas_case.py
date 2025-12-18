#!/usr/bin/env python3
"""MAS case study: visualize token-perturbation faithfulness for attribution methods.

This script matches the faithfulness evaluation logic implemented in:
  - evaluations/faithfulness.py
  - llm_attr_eval.LLMAttributionEvaluator.faithfulness_test()

For a single example and a selected attribution method, we:
  1) Compute token-level attributions (Seq / Row / Recursive) over prompt tokens.
  2) Rank prompt tokens by attribution mass.
  3) Iteratively perturb the prompt by replacing one token at a time with PAD tokens.
  4) Score the model as sum log p(generation + EOS | prompt) under the chat template.
  5) Compute RISE / MAS / RISE+AP (AUCs) and visualize the perturbation impact as token heatmaps.

Outputs JSON + HTML to exp/case_study/out/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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

# Avoid optional vision deps when importing transformers.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_TRANSFORMS", "1")


def _stub_torchvision() -> None:
    """Provide minimal torchvision stubs so transformers imports succeed without torchvision."""

    if "torchvision" in sys.modules:
        return

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


def _stub_timm() -> None:
    """Provide minimal timm stubs to avoid optional vision deps."""

    if "timm" in sys.modules:
        return

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


def _stub_gemma3n() -> None:
    """Stub Gemma3n config module if transformers tries to import it."""

    if "transformers.models.gemma3n.configuration_gemma3n" in sys.modules:
        return

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


_stub_torchvision()
_stub_timm()
_stub_gemma3n()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import transformers  # noqa: E402

# Provide light stubs if Longformer classes are unavailable; we don't use them here.
if not hasattr(transformers, "LongformerTokenizer"):
    class _DummyLongformerTokenizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("LongformerTokenizer stubbed; install full transformers if needed.")
    transformers.LongformerTokenizer = _DummyLongformerTokenizer
if not hasattr(transformers, "LongformerForMaskedLM"):
    class _DummyLongformerForMaskedLM:
        def __init__(self, *args, **kwargs):
            raise ImportError("LongformerForMaskedLM stubbed; install full transformers if needed.")
    transformers.LongformerForMaskedLM = _DummyLongformerForMaskedLM

from exp.case_study import viz  # noqa: E402
from exp.exp2 import dataset_utils as ds_utils  # noqa: E402
from shared_utils import DEFAULT_PROMPT_TEMPLATE  # noqa: E402

import llm_attr  # noqa: E402


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


def _pick_lrp_dtype() -> torch.dtype:
    """Prefer bf16 for AttnLRP to avoid fp16 gradient underflow."""

    if torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def load_model(model_name: str, device: str, *, torch_dtype: torch.dtype) -> Tuple[Any, Any]:
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_idx = int(device.split(":")[1])
        except Exception:
            gpu_idx = 0
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": gpu_idx},
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_example(dataset: str, index: int, data_root: Path) -> Tuple[ds_utils.CachedExample, str]:
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


def make_output_stem(dataset_name: str, index: int, method: str) -> str:
    safe_name = dataset_name.replace("/", "_").replace(" ", "_")
    return f"mas_case_{method}_{safe_name}_idx{index}"


def format_prompt(tokenizer: Any, prompt: str) -> str:
    modified_prompt = DEFAULT_PROMPT_TEMPLATE.format(context=prompt, query="")
    formatted_prompt = [{"role": "user", "content": modified_prompt}]
    return tokenizer.apply_chat_template(
        formatted_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def compute_logprob_response_given_prompt(model: Any, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
    """Compute log-probabilities of response_ids given prompt_ids.

    Shapes:
      prompt_ids: [B, N]
      response_ids: [B, M]
      returns: [B, M]
    """
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, N+M, V]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    response_start = int(prompt_ids.shape[1])
    logits_for_response = log_probs[:, response_start - 1 : -1, :]  # [B, M, V]
    gathered = logits_for_response.gather(2, response_ids.unsqueeze(-1))
    return gathered.squeeze(-1)


@torch.inference_mode()
def score_prompt_ids_with_generation(model: Any, *, prompt_ids: torch.Tensor, generation_ids: torch.Tensor) -> float:
    return float(compute_logprob_response_given_prompt(model, prompt_ids, generation_ids).sum().detach().cpu().item())


@torch.inference_mode()
def _ensure_pad_token_id(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id; cannot define baseline token.")
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id)


def _find_subsequence_start(haystack: torch.Tensor, needle: torch.Tensor) -> Optional[int]:
    if haystack.ndim != 1 or needle.ndim != 1:
        raise ValueError("Expected 1D tensors for subsequence matching.")
    if needle.numel() == 0:
        return 0
    hay_len = int(haystack.numel())
    needle_len = int(needle.numel())
    if needle_len > hay_len:
        return None
    for i in range(hay_len - needle_len + 1):
        if torch.equal(haystack[i : i + needle_len], needle):
            return i
    return None


def decode_text_into_tokens(tokenizer: Any, text: str) -> List[str]:
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = list(encoding["offset_mapping"])
    tokens: List[str] = []
    for start, end in offsets:
        tokens.append(text[start:end])
    return tokens


def auc(arr: np.ndarray) -> float:
    return float((arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1))


def mas_trace(
    model: Any,
    tokenizer: Any,
    *,
    attribution: torch.Tensor,
    prompt: str,
    generation: str,
) -> Dict[str, Any]:
    """Return a token-level faithfulness trace (RISE/MAS/RISE+AP) plus per-token deltas."""

    pad_token_id = _ensure_pad_token_id(tokenizer)

    user_prompt = " " + prompt
    formatted = format_prompt(tokenizer, user_prompt)
    formatted_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids
    user_ids = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    user_start = _find_subsequence_start(formatted_ids[0], user_ids[0])
    if user_start is None:
        raise RuntimeError("Failed to locate user prompt token span inside formatted chat prompt.")

    prompt_ids = formatted_ids.to(model.device)
    prompt_ids_perturbed = prompt_ids.clone()
    gen_ids = tokenizer(
        generation + (tokenizer.eos_token or ""),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)

    attr_cpu = attribution.detach().cpu()
    w = attr_cpu.sum(0)
    sorted_attr_indices = torch.argsort(w, descending=True)
    attr_sum = float(w.sum().item())

    P = int(w.numel())
    if int(user_ids.shape[1]) != P:
        raise ValueError(
            "Prompt-side attribution length does not match tokenized user prompt length: "
            f"attr P={P}, user_prompt P={int(user_ids.shape[1])}."
        )

    scores = np.zeros(P + 1, dtype=np.float64)
    density = np.zeros(P + 1, dtype=np.float64)

    scores[0] = score_prompt_ids_with_generation(model, prompt_ids=prompt_ids_perturbed, generation_ids=gen_ids)
    density[0] = 1.0

    if attr_sum <= 0:
        density = np.linspace(1.0, 0.0, P + 1)

    for step, idx_t in enumerate(sorted_attr_indices):
        idx = int(idx_t.item())
        prompt_ids_perturbed[0, user_start + idx] = pad_token_id

        scores[step + 1] = score_prompt_ids_with_generation(model, prompt_ids=prompt_ids_perturbed, generation_ids=gen_ids)
        if attr_sum > 0:
            dec = float(w[idx].item()) / attr_sum
            density[step + 1] = density[step] - dec

    min_normalized_pred = 1.0
    normalized_model_response = scores.copy()
    for i in range(len(scores)):
        normalized_pred = (normalized_model_response[i] - scores[-1]) / (abs(scores[0] - scores[-1]))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, float(normalized_pred))
        normalized_model_response[i] = min_normalized_pred

    alignment_penalty = np.abs(normalized_model_response - density)
    corrected_scores = normalized_model_response + alignment_penalty
    corrected_scores = corrected_scores.clip(0, 1)
    corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))
    if np.isnan(corrected_scores).any():
        corrected_scores = np.linspace(1, 0, len(scores))

    rise = auc(normalized_model_response)
    mas = auc(corrected_scores)
    rise_ap = auc(normalized_model_response + alignment_penalty)

    per_token_delta = np.zeros(P, dtype=np.float64)
    for step, idx_t in enumerate(sorted_attr_indices):
        idx = int(idx_t.item())
        per_token_delta[idx] = scores[step] - scores[step + 1]

    if attr_sum > 0:
        attr_weights = (w.numpy() / (attr_sum + 1e-12)).astype(np.float64)
    else:
        attr_weights = np.zeros(P, dtype=np.float64)

    return {
        "num_tokens": P,
        "sorted_attr_indices": [int(i.item()) for i in sorted_attr_indices],
        "scores_raw": scores.tolist(),
        "density": density.tolist(),
        "normalized_model_response": normalized_model_response.tolist(),
        "alignment_penalty": alignment_penalty.tolist(),
        "corrected_scores": corrected_scores.tolist(),
        "token_deltas_raw": per_token_delta.tolist(),
        "attr_weights": attr_weights.tolist(),
        "metrics": {"RISE": rise, "MAS": mas, "RISE+AP": rise_ap},
    }


def transform_values(values: Sequence[float], transform: str) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if transform == "positive":
        arr = np.clip(arr, 0.0, None)
    elif transform == "abs":
        arr = np.abs(arr)
    elif transform == "signed":
        pass
    else:
        raise ValueError(f"Unsupported score_transform={transform!r}; expected 'positive', 'abs', or 'signed'.")
    return arr.tolist()


def compute_method_attribution(
    method: str,
    example: ds_utils.CachedExample,
    model: Any,
    tokenizer: Any,
    *,
    n_hops: int,
    sink_span: Optional[Tuple[int, int]],
    thinking_span: Optional[Tuple[int, int]],
    chunk_tokens: int,
    sink_chunk_tokens: int,
) -> Tuple[str, Any, llm_attr.LLMAttributionResult]:
    prompt = example.prompt
    target = example.target

    if method == "ifr":
        if sink_span is None:
            raise ValueError("IFR requires sink_span (use dataset sink_span or pass --sink_span).")
        attributor = llm_attr.LLMIFRAttribution(model, tokenizer, chunk_tokens=chunk_tokens, sink_chunk_tokens=sink_chunk_tokens)
        result = attributor.calculate_ifr_span(prompt, target=target, span=sink_span)
        return "IFR (ifr_span)", attributor, result

    if method in ("ft", "ft_ifr"):
        attributor = llm_attr.LLMIFRAttribution(model, tokenizer, chunk_tokens=chunk_tokens, sink_chunk_tokens=sink_chunk_tokens)
        result = attributor.calculate_ifr_multi_hop(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=int(n_hops),
        )
        return "FT-IFR (ifr_multi_hop)", attributor, result

    if method == "attnlrp":
        attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        if sink_span is None:
            result = attributor.calculate_attnlrp_aggregated(prompt, target=target)
            return "AttnLRP (attnlrp_aggregated)", attributor, result

        sink_start, sink_end = int(sink_span[0]), int(sink_span[1])
        aggregate = attributor.calculate_attnlrp_span_aggregate(
            prompt,
            target=target,
            sink_start=sink_start,
            sink_end=sink_end,
            normalize_weights=True,
            score_mode="max",
        )

        gen_len = len(aggregate.generation_tokens)
        user_prompt_len = len(aggregate.user_prompt_tokens)
        expected_len = user_prompt_len + gen_len
        score_array = torch.full((gen_len, expected_len), torch.nan, dtype=torch.float32)
        for step in range(gen_len):
            gen_pos = user_prompt_len + step
            score_array[step, :gen_pos] = aggregate.token_importance_total[:gen_pos]

        metadata = {
            "method": "attnlrp_span_aggregate_wrapped",
            "sink_span": sink_span,
            "aggregate": aggregate,
        }
        result = llm_attr.LLMAttributionResult(
            tokenizer,
            score_array,
            aggregate.user_prompt_tokens,
            aggregate.generation_tokens,
            all_tokens=aggregate.all_tokens,
            metadata=metadata,
        )
        return "AttnLRP (attnlrp_span_aggregate)", attributor, result

    if method == "ft_attnlrp":
        attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        result = attributor.calculate_attnlrp_aggregated_multi_hop(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=int(n_hops),
        )
        return "FT-AttnLRP (attnlrp_aggregated_multi_hop)", attributor, result

    raise ValueError(f"Unsupported method={method!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MAS case study (faithfulness perturbation visualization)")
    parser.add_argument("--dataset", type=str, default="exp/exp2/data/morehopqa.jsonl", help="Dataset name or JSONL path.")
    parser.add_argument("--data_root", type=str, default="exp/exp2/data", help="Cache root for dataset names.")
    parser.add_argument("--index", type=int, default=0, help="Sample index (supports negative for reverse).")
    parser.add_argument("--method", type=str, choices=["ifr", "ft", "ft_ifr", "attnlrp", "ft_attnlrp"], default="ft")
    parser.add_argument("--model", type=str, default="qwen-8B", help="HF repo id (ignored if --model_path set).")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path to override --model.")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA spec (e.g., '0' or '0,1').")
    parser.add_argument("--cuda_num", type=int, default=0, help="Fallback GPU index when --cuda unset.")
    parser.add_argument("--n_hops", type=int, default=1, help="Number of hops for multi-hop methods.")
    parser.add_argument("--sink_span", type=int, nargs=2, default=None, help="Optional sink span over generation tokens.")
    parser.add_argument("--thinking_span", type=int, nargs=2, default=None, help="Optional thinking span over generation tokens.")
    parser.add_argument("--chunk_tokens", type=int, default=128, help="IFR chunk size.")
    parser.add_argument("--sink_chunk_tokens", type=int, default=32, help="IFR sink chunk size.")
    parser.add_argument("--score_transform", type=str, choices=["positive", "abs", "signed"], default="positive")
    parser.add_argument("--output_dir", type=str, default="exp/case_study/out", help="Where to write HTML/JSON artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.cuda, args.cuda_num)
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(f"[info] CUDA_VISIBLE_DEVICES={visible!r} torch.cuda.device_count()={torch.cuda.device_count()} device={device}")

    method_key = "ft" if args.method == "ft_ifr" else args.method
    torch_dtype = _pick_lrp_dtype() if method_key in ("attnlrp", "ft_attnlrp") else torch.float16

    model_name = args.model_path if args.model_path is not None else args.model
    model, tokenizer = load_model(model_name, device, torch_dtype=torch_dtype)

    example, ds_name = load_example(args.dataset, args.index, Path(args.data_root))

    sink_span = tuple(args.sink_span) if args.sink_span is not None else tuple(example.sink_span) if example.sink_span else None
    thinking_span = (
        tuple(args.thinking_span)
        if args.thinking_span is not None
        else tuple(example.thinking_span) if example.thinking_span else None
    )

    method_label, attributor, attr_result = compute_method_attribution(
        method_key,
        example,
        model,
        tokenizer,
        n_hops=args.n_hops,
        sink_span=sink_span,
        thinking_span=thinking_span,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
    )

    indices_to_explain = example.indices_to_explain or example.sink_span
    if not (isinstance(indices_to_explain, list) and len(indices_to_explain) == 2):
        raise ValueError("MAS case study requires token-span indices_to_explain=[start_tok,end_tok] (e.g. sink_span).")
    seq_attr, row_attr, rec_attr = attr_result.get_all_token_attrs(indices_to_explain)

    prompt_tokens = decode_text_into_tokens(tokenizer, " " + example.prompt)
    generation_text = example.target if example.target is not None else (getattr(attributor, "generation", None) or "")

    variant_specs = [
        ("seq", "Seq attribution", seq_attr),
        ("row", "Row attribution", row_attr),
        ("recursive", "Recursive attribution", rec_attr),
    ]

    formatted = format_prompt(tokenizer, " " + example.prompt)
    prompt_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    gen_ids = tokenizer(
        generation_text + (tokenizer.eos_token or ""),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)
    base_score = score_prompt_ids_with_generation(model, prompt_ids=prompt_ids, generation_ids=gen_ids)

    panels_raw: List[Dict[str, Any]] = []
    panels_display: List[Dict[str, Any]] = []

    for variant_key, variant_label, variant_attr in variant_specs:
        prompt_len = int(seq_attr.shape[1] - seq_attr.shape[0])  # cols=(P+G), rows=G
        attr_prompt = variant_attr[:, :prompt_len]
        trace = mas_trace(
            model,
            tokenizer,
            attribution=attr_prompt.to(device="cpu"),
            prompt=example.prompt,
            generation=generation_text,
        )
        trace["variant"] = variant_key
        trace["variant_label"] = variant_label

        panel_raw = {
            "variant": variant_key,
            "variant_label": variant_label,
            "metrics": trace.get("metrics"),
            "sorted_attr_indices": trace.get("sorted_attr_indices"),
            "attr_weights": trace.get("attr_weights"),
            "token_deltas_raw": trace.get("token_deltas_raw"),
            "mas_trace": trace,
        }
        panels_raw.append(panel_raw)

        panel_display = {
            "variant": variant_key,
            "variant_label": variant_label,
            "metrics": trace.get("metrics"),
            "sorted_attr_indices": trace.get("sorted_attr_indices"),
            "attr_weights": transform_values(trace.get("attr_weights", []), "signed"),
            "token_deltas_raw": transform_values(trace.get("token_deltas_raw", []), args.score_transform),
        }
        panels_display.append(panel_display)

    case_meta: Dict[str, Any] = {
        "dataset": ds_name,
        "index": args.index,
        "mode": "mas",
        "attr_method": method_key,
        "attr_method_label": method_label,
        "sink_span": sink_span,
        "thinking_span": thinking_span,
        "n_hops": int(args.n_hops),
        "score_transform": args.score_transform,
        "base_score": float(base_score),
    }

    record = {
        "meta": case_meta,
        "prompt": example.prompt,
        "target": example.target,
        "generation": generation_text,
        "prompt_tokens": prompt_tokens,
        "panels": panels_raw,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = make_output_stem(ds_name, args.index, method_key)
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    html = viz.render_mas_token_html(
        case_meta,
        prompt_tokens=prompt_tokens,
        panels=panels_display,
        generation=generation_text,
    )
    html_path.write_text(html, encoding="utf-8")

    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {html_path}")


if __name__ == "__main__":
    main()
