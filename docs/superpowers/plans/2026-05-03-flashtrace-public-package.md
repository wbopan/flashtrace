# FlashTrace Public Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable `flashtrace` package with a stable Python API, CLI tracing command, JSON export, HTML heatmap export, README quickstart, and CPU smoke tests.

**Architecture:** Create a package-first structure while preserving temporary root compatibility wrappers for existing experiment scripts. Move the IFR implementation and attribution engines into `flashtrace/`, wrap them with `FlashTrace` and `TraceResult`, then expose a CLI and public examples.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Accelerate, NumPy, tqdm, argparse, pytest.

---

## File Structure

Create or modify these files:

- Create: `flashtrace/__init__.py` for public exports.
- Create: `flashtrace/core.py` from `ifr_core.py`.
- Create: `flashtrace/shared_utils.py` from `shared_utils.py`.
- Create: `flashtrace/lrp_rules.py` from `lrp_rules.py`.
- Create: `flashtrace/lrp_patches.py` from `lrp_patches.py`.
- Create: `flashtrace/attribution.py` from `llm_attr.py`.
- Create: `flashtrace/improved.py` from `ft_ifr_improve.py`.
- Create: `flashtrace/result.py` for `TokenScore` and `TraceResult`.
- Create: `flashtrace/viz.py` for standalone HTML token heatmaps.
- Create: `flashtrace/tracer.py` for the `FlashTrace` facade.
- Create: `flashtrace/model_io.py` for Hugging Face loading helpers.
- Create: `flashtrace/cli.py` for `flashtrace trace`.
- Create: `flashtrace/baselines/__init__.py`.
- Create: `flashtrace/baselines/attnlrp.py`.
- Modify: `ifr_core.py`, `shared_utils.py`, `lrp_rules.py`, `lrp_patches.py`, `llm_attr.py`, `ft_ifr_improve.py` into root compatibility wrappers.
- Modify: `pyproject.toml` package metadata and console script.
- Modify: `.gitignore` generated artifact rules.
- Create: `README.md`.
- Create: `LICENSE`.
- Create: `examples/quickstart.py`.
- Create: `tests/helpers.py`.
- Create: `tests/test_imports.py`.
- Create: `tests/test_core_recompute.py`.
- Create: `tests/test_result.py`.
- Create: `tests/test_tracer.py`.
- Create: `tests/test_cli.py`.
- Delete: `model_generation.py`.

## Task 1: Package Metadata And Skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `flashtrace/__init__.py`
- Create: `flashtrace/tracer.py`
- Create: `flashtrace/result.py`
- Create: `flashtrace/model_io.py`
- Create: `flashtrace/cli.py`
- Create: `flashtrace/baselines/__init__.py`
- Create: `flashtrace/baselines/attnlrp.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write the failing public import test**

Create `tests/test_imports.py`:

```python
def test_public_imports():
    import flashtrace

    assert flashtrace.FlashTrace.__name__ == "FlashTrace"
    assert flashtrace.TraceResult.__name__ == "TraceResult"
    assert callable(flashtrace.load_model_and_tokenizer)
```

- [ ] **Step 2: Run the import test and see the expected failure**

Run:

```bash
uv run pytest tests/test_imports.py -q
```

Expected: pytest reports an import failure for `flashtrace`.

- [ ] **Step 3: Create package directories**

Run:

```bash
mkdir -p flashtrace/baselines tests
```

- [ ] **Step 4: Add minimal public package files**

Create `flashtrace/tracer.py`:

```python
class FlashTrace:
    """Public facade for FlashTrace attribution."""

    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.options = dict(kwargs)
```

Create `flashtrace/result.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceResult:
    """Public attribution result returned by FlashTrace."""

    prompt_tokens: list[str]
    generation_tokens: list[str]
    scores: list[float]
```

Create `flashtrace/model_io.py`:

```python
def load_model_and_tokenizer(*args, **kwargs):
    """Load a Hugging Face causal LM and tokenizer."""

    raise RuntimeError("load_model_and_tokenizer will be implemented in the model IO task.")
```

Create `flashtrace/cli.py`:

```python
def main(argv=None):
    """FlashTrace command-line entrypoint."""

    raise RuntimeError("CLI will be implemented in the CLI task.")
```

Create `flashtrace/baselines/__init__.py`:

```python
"""Baseline attribution methods for FlashTrace."""
```

Create `flashtrace/baselines/attnlrp.py`:

```python
"""AttnLRP baseline exports."""
```

Create `flashtrace/__init__.py`:

```python
"""FlashTrace: efficient multi-token attribution for reasoning LLMs."""

from .model_io import load_model_and_tokenizer
from .result import TraceResult
from .tracer import FlashTrace

__all__ = ["FlashTrace", "TraceResult", "load_model_and_tokenizer"]
```

- [ ] **Step 5: Update package metadata**

Replace `pyproject.toml` with:

```toml
[project]
name = "flashtrace"
version = "0.1.0"
description = "Efficient multi-token attribution for reasoning language models."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "accelerate>=1.11.0",
    "matplotlib>=3.6",
    "networkx>=3.3",
    "numpy>=2.0",
    "seaborn>=0.11",
    "spacy>=3.8",
    "torch>=2.5",
    "tqdm>=4.67",
    "transformers>=4.53",
    "wordfreq>=3.1.1",
]

[project.optional-dependencies]
baselines = [
    "bert-score>=0.3.13",
    "evaluate>=0.4.6",
    "sentence-transformers>=4.1.0",
]
eval = [
    "datasets>=2.21",
    "evaluate>=0.4.6",
]
dev = [
    "pytest>=8.0",
]

[project.scripts]
flashtrace = "flashtrace.cli:main"

[tool.setuptools.packages.find]
include = ["flashtrace*"]
```

- [ ] **Step 6: Run the import test**

Run:

```bash
uv run pytest tests/test_imports.py -q
```

Expected: `1 passed`.

- [ ] **Step 7: Commit**

Run:

```bash
git add pyproject.toml flashtrace tests/test_imports.py
git commit -m "feat: add flashtrace package skeleton"
```

## Task 2: Core IFR Migration

**Files:**
- Create: `flashtrace/core.py`
- Create: `flashtrace/shared_utils.py`
- Modify: `ifr_core.py`
- Modify: `shared_utils.py`
- Create: `tests/helpers.py`
- Create: `tests/test_core_recompute.py`

- [ ] **Step 1: Add the tiny-model test helper**

Create `tests/helpers.py`:

```python
from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast


def make_tiny_qwen2_model_and_tokenizer(
    *,
    n_layers: int = 3,
    d_model: int = 48,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    max_pos: int = 128,
):
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=500,
        hidden_size=d_model,
        intermediate_size=d_model * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=max_pos,
        use_sliding_window=False,
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()

    backend = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(500)}, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="t1", pad_token="t2")
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return model, tokenizer
```

- [ ] **Step 2: Write the failing core import smoke test**

Create `tests/test_core_recompute.py`:

```python
import torch

from flashtrace import core
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def test_core_metadata_and_weight_pack():
    model, _ = make_tiny_qwen2_model_and_tokenizer()

    metadata = core.extract_model_metadata(model)
    weight_pack = core.build_weight_pack(metadata, next(model.parameters()).dtype)

    assert metadata.n_layers == 3
    assert metadata.n_heads_q == 4
    assert metadata.n_kv_heads == 2
    assert len(weight_pack) == 3
    assert torch.is_tensor(weight_pack[0]["v_w"])
```

- [ ] **Step 3: Run the core smoke test and see the expected failure**

Run:

```bash
uv run pytest tests/test_core_recompute.py::test_core_metadata_and_weight_pack -q
```

Expected: pytest reports missing `flashtrace.core`.

- [ ] **Step 4: Copy the IFR core into the package**

Run:

```bash
cp ifr_core.py flashtrace/core.py
```

- [ ] **Step 5: Copy shared utilities into the package**

Run:

```bash
cp shared_utils.py flashtrace/shared_utils.py
```

- [ ] **Step 6: Replace root `ifr_core.py` with a compatibility wrapper**

Replace `ifr_core.py` with:

```python
"""Compatibility wrapper for package-era imports."""

from flashtrace.core import *  # noqa: F401,F403
```

- [ ] **Step 7: Replace root `shared_utils.py` with a compatibility wrapper**

Replace `shared_utils.py` with:

```python
"""Compatibility wrapper for package-era imports."""

from flashtrace.shared_utils import *  # noqa: F401,F403
```

- [ ] **Step 8: Run the core smoke test**

Run:

```bash
uv run pytest tests/test_core_recompute.py::test_core_metadata_and_weight_pack -q
```

Expected: `1 passed`.

- [ ] **Step 9: Commit**

Run:

```bash
git add flashtrace/core.py flashtrace/shared_utils.py ifr_core.py shared_utils.py tests/helpers.py tests/test_core_recompute.py
git commit -m "feat: move IFR core into package"
```

## Task 3: Attribution Engine Migration

**Files:**
- Create: `flashtrace/lrp_rules.py`
- Create: `flashtrace/lrp_patches.py`
- Create: `flashtrace/attribution.py`
- Create: `flashtrace/improved.py`
- Modify: `lrp_rules.py`
- Modify: `lrp_patches.py`
- Modify: `llm_attr.py`
- Modify: `ft_ifr_improve.py`
- Modify: `flashtrace/baselines/attnlrp.py`
- Test: `tests/test_core_recompute.py`

- [ ] **Step 1: Extend the recompute test with package attribution paths**

Append to `tests/test_core_recompute.py`:

```python
from flashtrace.attribution import LLMIFRAttribution


def test_package_attribution_recompute_matches_stored_attention():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    prompt = "t10 t20 t30 t40"
    target = "t60 t70"

    stored = LLMIFRAttribution(model, tokenizer, recompute_attention=False).calculate_ifr_span(prompt, target)
    recomputed = LLMIFRAttribution(model, tokenizer, recompute_attention=True).calculate_ifr_span(prompt, target)

    diff = (stored.attribution_matrix - recomputed.attribution_matrix).abs().max().item()
    assert diff < 1e-5
```

- [ ] **Step 2: Run the package attribution test and see the expected failure**

Run:

```bash
uv run pytest tests/test_core_recompute.py::test_package_attribution_recompute_matches_stored_attention -q
```

Expected: pytest reports missing `flashtrace.attribution`.

- [ ] **Step 3: Copy LRP helpers and attribution engines into the package**

Run:

```bash
cp lrp_rules.py flashtrace/lrp_rules.py
cp lrp_patches.py flashtrace/lrp_patches.py
cp llm_attr.py flashtrace/attribution.py
cp ft_ifr_improve.py flashtrace/improved.py
```

- [ ] **Step 4: Update imports in `flashtrace/attribution.py`**

Edit package-local imports to this form:

```python
from .core import (
    IFRParameters,
    ModelMetadata,
    attach_hooks,
    build_weight_pack,
    compute_ifr_for_all_positions,
    compute_ifr_sentence_aggregate,
    compute_multi_hop_ifr,
    extract_model_metadata,
)
from .shared_utils import (
    DEFAULT_GENERATE_KWARGS,
    DEFAULT_PROMPT_TEMPLATE,
    create_sentences,
    create_sentence_masks,
)
from .lrp_patches import lrp_context, detect_model_type
```

- [ ] **Step 5: Update imports in `flashtrace/lrp_patches.py`**

Edit the LRP helper import to:

```python
from .lrp_rules import stop_gradient, divide_gradient, identity_rule_implicit
```

- [ ] **Step 6: Update imports in `flashtrace/improved.py`**

Edit the top-level package imports to:

```python
from . import attribution as llm_attr
from .core import IFRAggregate, MultiHopIFRResult, compute_ifr_sentence_aggregate
```

- [ ] **Step 7: Replace root compatibility modules**

Replace `lrp_rules.py` with:

```python
"""Compatibility wrapper for package-era imports."""

from flashtrace.lrp_rules import *  # noqa: F401,F403
```

Replace `lrp_patches.py` with:

```python
"""Compatibility wrapper for package-era imports."""

from flashtrace.lrp_patches import *  # noqa: F401,F403
```

Replace `llm_attr.py` with:

```python
"""Compatibility wrapper for package-era imports."""

from flashtrace.attribution import *  # noqa: F401,F403
```

Replace `ft_ifr_improve.py` with:

```python
"""Compatibility wrapper for package-era imports."""

from flashtrace.improved import *  # noqa: F401,F403
```

- [ ] **Step 8: Export the AttnLRP baseline**

Replace `flashtrace/baselines/attnlrp.py` with:

```python
"""AttnLRP baseline API."""

from flashtrace.attribution import AttnLRPSpanAggregate, LLMLRPAttribution, MultiHopAttnLRPResult
from flashtrace.lrp_patches import detect_model_type, lrp_context

__all__ = [
    "AttnLRPSpanAggregate",
    "LLMLRPAttribution",
    "MultiHopAttnLRPResult",
    "detect_model_type",
    "lrp_context",
]
```

Replace `flashtrace/baselines/__init__.py` with:

```python
"""Baseline attribution methods for FlashTrace."""

from .attnlrp import LLMLRPAttribution

__all__ = ["LLMLRPAttribution"]
```

- [ ] **Step 9: Run attribution migration tests**

Run:

```bash
uv run pytest tests/test_core_recompute.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 10: Run a root compatibility import check**

Run:

```bash
uv run python -c "import ifr_core, llm_attr, ft_ifr_improve; print(llm_attr.LLMIFRAttribution.__name__)"
```

Expected: prints `LLMIFRAttribution`.

- [ ] **Step 11: Commit**

Run:

```bash
git add flashtrace lrp_rules.py lrp_patches.py llm_attr.py ft_ifr_improve.py tests/test_core_recompute.py
git commit -m "feat: move attribution engines into package"
```

## Task 4: TraceResult And HTML Heatmap

**Files:**
- Modify: `flashtrace/result.py`
- Create: `flashtrace/viz.py`
- Create: `tests/test_result.py`

- [ ] **Step 1: Write result object tests**

Create `tests/test_result.py`:

```python
import json

from flashtrace.result import TokenScore, TraceResult


def make_result():
    return TraceResult(
        prompt_tokens=[" alpha", " beta", " gamma"],
        generation_tokens=[" answer"],
        scores=[0.2, 0.7, 0.1],
        per_hop_scores=[[0.1, 0.4, 0.0], [0.1, 0.3, 0.1]],
        thinking_ratios=[0.5, 0.2],
        output_span=(0, 0),
        reasoning_span=(0, 0),
        method="flashtrace",
        metadata={"model": "tiny"},
    )


def test_topk_inputs_sorted():
    result = make_result()

    top = result.topk_inputs(2)

    assert top == [
        TokenScore(index=1, token=" beta", score=0.7),
        TokenScore(index=0, token=" alpha", score=0.2),
    ]


def test_to_dict_is_json_serializable():
    result = make_result()

    payload = result.to_dict()

    assert payload["method"] == "flashtrace"
    assert payload["top_inputs"][0]["token"] == " beta"
    json.dumps(payload)


def test_to_dict_sanitizes_tensor_metadata():
    import torch

    result = TraceResult(
        prompt_tokens=[" alpha"],
        generation_tokens=[" answer"],
        scores=[1.0],
        metadata={"tensor": torch.tensor([1.0, 2.0]), "object": object()},
    )

    payload = result.to_dict()

    assert payload["metadata"]["tensor"] == [1.0, 2.0]
    assert isinstance(payload["metadata"]["object"], str)
    json.dumps(payload)


def test_json_and_html_export(tmp_path):
    result = make_result()
    json_path = tmp_path / "trace.json"
    html_path = tmp_path / "trace.html"

    result.to_json(json_path)
    result.to_html(html_path)

    assert json_path.read_text(encoding="utf-8").startswith("{")
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html
    assert " beta" in html
```

- [ ] **Step 2: Run result tests and see the expected failure**

Run:

```bash
uv run pytest tests/test_result.py -q
```

Expected: pytest reports missing `TokenScore` or missing methods.

- [ ] **Step 3: Implement `TraceResult`**

Replace `flashtrace/result.py` with:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenScore:
    index: int
    token: str
    score: float


@dataclass(frozen=True)
class TraceResult:
    """Public attribution result returned by FlashTrace."""

    prompt_tokens: list[str]
    generation_tokens: list[str]
    scores: list[float]
    per_hop_scores: list[list[float]] = field(default_factory=list)
    thinking_ratios: list[float] = field(default_factory=list)
    output_span: tuple[int, int] | None = None
    reasoning_span: tuple[int, int] | None = None
    method: str = "flashtrace"
    metadata: dict[str, Any] = field(default_factory=dict)

    def topk_inputs(self, k: int = 20) -> list[TokenScore]:
        limit = max(0, int(k))
        items = [
            TokenScore(index=i, token=tok, score=float(score))
            for i, (tok, score) in enumerate(zip(self.prompt_tokens, self.scores))
        ]
        items.sort(key=lambda item: item.score, reverse=True)
        return items[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "prompt_tokens": list(self.prompt_tokens),
            "generation_tokens": list(self.generation_tokens),
            "scores": [float(x) for x in self.scores],
            "per_hop_scores": [[float(x) for x in row] for row in self.per_hop_scores],
            "thinking_ratios": [float(x) for x in self.thinking_ratios],
            "output_span": list(self.output_span) if self.output_span is not None else None,
            "reasoning_span": list(self.reasoning_span) if self.reasoning_span is not None else None,
            "top_inputs": [asdict(item) for item in self.topk_inputs()],
            "metadata": _jsonable(self.metadata),
        }

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_html(self, path: str | Path) -> None:
        from .viz import render_trace_html

        target = Path(path)
        target.write_text(render_trace_html(self), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            return repr(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)
```

- [ ] **Step 4: Implement the standalone HTML renderer**

Create `flashtrace/viz.py`:

```python
from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import TraceResult


def _score_color(score: float, max_score: float) -> str:
    if max_score <= 0.0:
        return "rgba(245,245,245,0.75)"
    ratio = min(1.0, abs(float(score)) / (max_score + 1e-12))
    red = 255
    green = int(246 - 105 * ratio)
    blue = int(226 - 170 * ratio)
    alpha = 0.22 + 0.58 * ratio
    return f"rgba({red},{green},{blue},{alpha:.3f})"


def _render_token_row(tokens: list[str], scores: list[float]) -> str:
    max_score = max((abs(float(x)) for x in scores), default=0.0)
    spans = []
    for index, token in enumerate(tokens):
        score = float(scores[index]) if index < len(scores) else 0.0
        color = _score_color(score, max_score)
        spans.append(
            "<span class='tok' "
            f"title='idx={index} score={score:.6f}' "
            f"style='background:{color}'>{escape(token)}</span>"
        )
    return "".join(spans)


def render_trace_html(result: "TraceResult") -> str:
    top_rows = "\n".join(
        f"<tr><td>{item.index}</td><td><code>{escape(item.token)}</code></td><td>{item.score:.6f}</td></tr>"
        for item in result.topk_inputs(20)
    )
    hop_sections = []
    for hop_index, hop_scores in enumerate(result.per_hop_scores):
        hop_sections.append(
            f"<section><h2>Hop {hop_index}</h2><div class='tokens'>{_render_token_row(result.prompt_tokens, hop_scores)}</div></section>"
        )
    hop_html = "\n".join(hop_sections)
    metadata = escape(str(result.metadata))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FlashTrace</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #151515; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 24px 0; }}
    .tokens {{ line-height: 2.2; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .tok {{ display: inline-block; margin: 2px; padding: 2px 4px; border-radius: 4px; white-space: pre-wrap; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    td, th {{ border-bottom: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    .meta {{ color: #555; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>FlashTrace</h1>
  <p class="meta">method={escape(result.method)} output_span={escape(str(result.output_span))} reasoning_span={escape(str(result.reasoning_span))}</p>
  <section>
    <h2>Prompt Attribution</h2>
    <div class="tokens">{_render_token_row(result.prompt_tokens, result.scores)}</div>
  </section>
  {hop_html}
  <section>
    <h2>Top Input Tokens</h2>
    <table><thead><tr><th>Index</th><th>Token</th><th>Score</th></tr></thead><tbody>{top_rows}</tbody></table>
  </section>
  <section>
    <h2>Metadata</h2>
    <pre>{metadata}</pre>
  </section>
</body>
</html>
"""
```

- [ ] **Step 5: Run result tests**

Run:

```bash
uv run pytest tests/test_result.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

Run:

```bash
git add flashtrace/result.py flashtrace/viz.py tests/test_result.py
git commit -m "feat: add trace result exports"
```

## Task 5: FlashTrace Facade

**Files:**
- Modify: `flashtrace/tracer.py`
- Modify: `flashtrace/__init__.py`
- Create: `tests/test_tracer.py`

- [ ] **Step 1: Write tracer API tests**

Create `tests/test_tracer.py`:

```python
from flashtrace import FlashTrace, TraceResult
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def test_flashtrace_trace_returns_public_result():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    tracer = FlashTrace(model, tokenizer, chunk_tokens=16, sink_chunk_tokens=4, recompute_attention=True)

    result = tracer.trace(
        prompt="t10 t20 t30 t40",
        target="t60 t70 t80",
        output_span=(1, 2),
        reasoning_span=(0, 1),
        hops=1,
    )

    assert isinstance(result, TraceResult)
    assert result.method == "flashtrace"
    assert len(result.prompt_tokens) > 0
    assert len(result.scores) == len(result.prompt_tokens)
    assert result.output_span == (1, 2)
    assert result.reasoning_span == (0, 1)


def test_ifr_span_method_returns_public_result():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    tracer = FlashTrace(model, tokenizer, chunk_tokens=16, sink_chunk_tokens=4, recompute_attention=True)

    result = tracer.trace(
        prompt="t10 t20 t30 t40",
        target="t60 t70",
        output_span=(0, 1),
        method="ifr-span",
    )

    assert result.method == "ifr-span"
    assert len(result.scores) == len(result.prompt_tokens)
```

- [ ] **Step 2: Run tracer tests and see the expected failure**

Run:

```bash
uv run pytest tests/test_tracer.py -q
```

Expected: pytest reports missing `trace`.

- [ ] **Step 3: Implement result adaptation helpers and facade**

Replace `flashtrace/tracer.py` with:

```python
from __future__ import annotations

from typing import Any, Literal

import torch

from .attribution import LLMIFRAttribution, LLMAttributionResult
from .improved import LLMIFRAttributionBoth
from .result import TraceResult

TraceMethod = Literal["flashtrace", "ifr-span", "ifr-matrix"]


def _to_float_list(values: Any) -> list[float]:
    if torch.is_tensor(values):
        values = values.detach().cpu().to(dtype=torch.float32).tolist()
    return [float(x) for x in (values or [])]


class FlashTrace:
    """Public facade for FlashTrace attribution."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        chunk_tokens: int = 128,
        sink_chunk_tokens: int = 32,
        recompute_attention: bool = False,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_tokens = int(chunk_tokens)
        self.sink_chunk_tokens = int(sink_chunk_tokens)
        self.recompute_attention = bool(recompute_attention)
        self.generate_kwargs = generate_kwargs

    def trace(
        self,
        *,
        prompt: str,
        target: str | None = None,
        output_span: tuple[int, int] | None = None,
        reasoning_span: tuple[int, int] | None = None,
        hops: int = 1,
        method: TraceMethod = "flashtrace",
        renorm_threshold: float | None = None,
    ) -> TraceResult:
        if method == "flashtrace":
            engine = LLMIFRAttributionBoth(
                self.model,
                self.tokenizer,
                generate_kwargs=self.generate_kwargs,
                chunk_tokens=self.chunk_tokens,
                sink_chunk_tokens=self.sink_chunk_tokens,
                recompute_attention=self.recompute_attention,
            )
            raw = engine.calculate_ifr_multi_hop_both(
                prompt,
                target=target,
                sink_span=output_span,
                thinking_span=reasoning_span,
                n_hops=int(hops),
                renorm_threshold=renorm_threshold,
            )
        elif method == "ifr-span":
            engine = LLMIFRAttribution(
                self.model,
                self.tokenizer,
                generate_kwargs=self.generate_kwargs,
                chunk_tokens=self.chunk_tokens,
                sink_chunk_tokens=self.sink_chunk_tokens,
                recompute_attention=self.recompute_attention,
            )
            raw = engine.calculate_ifr_span(
                prompt,
                target=target,
                span=output_span,
                renorm_threshold=renorm_threshold,
            )
        elif method == "ifr-matrix":
            engine = LLMIFRAttribution(
                self.model,
                self.tokenizer,
                generate_kwargs=self.generate_kwargs,
                chunk_tokens=self.chunk_tokens,
                sink_chunk_tokens=self.sink_chunk_tokens,
                recompute_attention=self.recompute_attention,
            )
            raw = engine.calculate_ifr_for_all_positions_output_only(
                prompt,
                target=target,
                sink_span=output_span,
                renorm_threshold=renorm_threshold,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        return self._build_result(raw, method=method, output_span=output_span, reasoning_span=reasoning_span)

    def _build_result(
        self,
        raw: LLMAttributionResult,
        *,
        method: str,
        output_span: tuple[int, int] | None,
        reasoning_span: tuple[int, int] | None,
    ) -> TraceResult:
        prompt_tokens = list(raw.prompt_tokens)
        generation_tokens = list(raw.generation_tokens)
        prompt_len = len(prompt_tokens)
        metadata = dict(raw.metadata or {})
        if "method" not in metadata:
            metadata["method"] = method

        ifr_meta = metadata.get("ifr") if isinstance(metadata.get("ifr"), dict) else {}
        observation = ifr_meta.get("observation_projected") if isinstance(ifr_meta, dict) else None
        per_hop_projected = ifr_meta.get("per_hop_projected") if isinstance(ifr_meta, dict) else None

        if isinstance(observation, dict) and "sum" in observation:
            vector = _to_float_list(observation["sum"])
            scores = vector[:prompt_len]
        else:
            matrix = torch.nan_to_num(raw.attribution_matrix.detach().cpu().to(dtype=torch.float32), nan=0.0)
            if output_span is not None:
                start, end = output_span
                selected = matrix[int(start) : int(end) + 1, :prompt_len]
            else:
                selected = matrix[:, :prompt_len]
            scores = selected.mean(dim=0).tolist() if selected.numel() else [0.0 for _ in prompt_tokens]

        per_hop_scores: list[list[float]] = []
        if per_hop_projected:
            for hop_vector in per_hop_projected:
                per_hop_scores.append(_to_float_list(hop_vector)[:prompt_len])

        ratios = ifr_meta.get("thinking_ratios", []) if isinstance(ifr_meta, dict) else []
        return TraceResult(
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            scores=[float(x) for x in scores],
            per_hop_scores=per_hop_scores,
            thinking_ratios=_to_float_list(ratios),
            output_span=output_span,
            reasoning_span=reasoning_span,
            method=method,
            metadata=metadata,
        )
```

- [ ] **Step 4: Confirm public exports**

Keep `flashtrace/__init__.py` as:

```python
"""FlashTrace: efficient multi-token attribution for reasoning LLMs."""

from .model_io import load_model_and_tokenizer
from .result import TokenScore, TraceResult
from .tracer import FlashTrace

__all__ = ["FlashTrace", "TraceResult", "TokenScore", "load_model_and_tokenizer"]
```

- [ ] **Step 5: Run tracer tests**

Run:

```bash
uv run pytest tests/test_tracer.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Run package tests created so far**

Run:

```bash
uv run pytest tests/test_imports.py tests/test_core_recompute.py tests/test_result.py tests/test_tracer.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add flashtrace/tracer.py flashtrace/__init__.py tests/test_tracer.py
git commit -m "feat: add FlashTrace public facade"
```

## Task 6: Model IO And CLI

**Files:**
- Modify: `flashtrace/model_io.py`
- Modify: `flashtrace/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write CLI tests**

Create `tests/test_cli.py`:

```python
import pytest

from flashtrace.cli import main, parse_span


def test_parse_span():
    assert parse_span("3:8") == (3, 8)
    assert parse_span(None) is None


@pytest.mark.parametrize("value", ["3", "8:3", "a:b"])
def test_parse_span_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_span(value)


def test_cli_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "trace" in capsys.readouterr().out
```

- [ ] **Step 2: Run CLI tests and see the expected failure**

Run:

```bash
uv run pytest tests/test_cli.py -q
```

Expected: pytest reports missing `parse_span`.

- [ ] **Step 3: Implement model loading**

Replace `flashtrace/model_io.py` with:

```python
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype: str | torch.dtype = "auto") -> str | torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    value = str(dtype).lower()
    if value == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[value]


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
    **model_kwargs: Any,
):
    """Load a Hugging Face causal LM and matching tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_resolve_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
```

- [ ] **Step 4: Implement CLI**

Replace `flashtrace/cli.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .model_io import load_model_and_tokenizer
from .tracer import FlashTrace


def parse_span(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError("Span must use START:END format.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError("Span bounds must be integers.") from exc
    if start < 0 or end < start:
        raise ValueError("Span must satisfy 0 <= START <= END.")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flashtrace", description="Trace language model outputs with FlashTrace.")
    sub = parser.add_subparsers(dest="command")

    trace = sub.add_parser("trace", help="Run attribution for a prompt and target.")
    trace.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    trace.add_argument("--prompt", required=True, help="UTF-8 text file containing the prompt.")
    trace.add_argument("--target", help="UTF-8 text file containing the target response.")
    trace.add_argument("--output-span", help="Inclusive generation-token span START:END.")
    trace.add_argument("--reasoning-span", help="Inclusive generation-token span START:END.")
    trace.add_argument("--hops", type=int, default=1)
    trace.add_argument("--method", default="flashtrace", choices=["flashtrace", "ifr-span", "ifr-matrix"])
    trace.add_argument("--html", help="Write standalone HTML heatmap.")
    trace.add_argument("--json", help="Write JSON trace.")
    trace.add_argument("--device-map", default="auto")
    trace.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    trace.add_argument("--chunk-tokens", type=int, default=128)
    trace.add_argument("--sink-chunk-tokens", type=int, default=32)
    trace.add_argument("--recompute-attention", action="store_true")
    return parser


def _read_text(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def _run_trace(args: argparse.Namespace) -> int:
    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype=args.dtype)
    tracer = FlashTrace(
        model,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
        recompute_attention=args.recompute_attention,
    )
    result = tracer.trace(
        prompt=_read_text(args.prompt) or "",
        target=_read_text(args.target),
        output_span=parse_span(args.output_span),
        reasoning_span=parse_span(args.reasoning_span),
        hops=args.hops,
        method=args.method,
    )
    for item in result.topk_inputs(20):
        print(f"{item.index}\t{item.score:.6f}\t{item.token!r}")
    if args.json:
        result.to_json(args.json)
    if args.html:
        result.to_html(args.html)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "trace":
        return _run_trace(args)
    parser.print_help()
    return 0
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
uv run pytest tests/test_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 6: Verify console script metadata**

Run:

```bash
uv run flashtrace --help
```

Expected: help text includes `trace`.

- [ ] **Step 7: Commit**

Run:

```bash
git add flashtrace/model_io.py flashtrace/cli.py tests/test_cli.py
git commit -m "feat: add model loader and CLI"
```

## Task 7: README, Example, License, And Release Hygiene

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `examples/quickstart.py`
- Modify: `.gitignore`
- Delete: `model_generation.py`

- [ ] **Step 1: Create the quickstart example**

Create `examples/quickstart.py`:

```python
from __future__ import annotations

import argparse

from flashtrace import FlashTrace, load_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashTrace quickstart example.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--target", help="Target response text.")
    parser.add_argument("--output-span", default=None, help="Inclusive generation-token span START:END.")
    parser.add_argument("--reasoning-span", default=None, help="Inclusive generation-token span START:END.")
    parser.add_argument("--html", default="trace.html", help="Output HTML path.")
    return parser


def parse_span(value: str | None) -> tuple[int, int] | None:
    from flashtrace.cli import parse_span as parse_cli_span

    return parse_cli_span(value)


def main() -> int:
    args = build_parser().parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model)
    tracer = FlashTrace(model, tokenizer)
    trace = tracer.trace(
        prompt=args.prompt,
        target=args.target,
        output_span=parse_span(args.output_span),
        reasoning_span=parse_span(args.reasoning_span),
    )
    for item in trace.topk_inputs(10):
        print(f"{item.index}\t{item.score:.6f}\t{item.token!r}")
    trace.to_html(args.html)
    print(f"wrote {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add README**

Create `README.md`:

```markdown
# FlashTrace

FlashTrace is an efficient multi-token attribution toolkit for reasoning language models. It implements the method described in [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](https://arxiv.org/abs/2602.01914).

## Install

```bash
pip install -e .
```

## Python Quickstart

```python
from flashtrace import FlashTrace, load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-8B")
tracer = FlashTrace(model, tokenizer)

trace = tracer.trace(
    prompt="Context: Paris is the capital of France.\nQuestion: What is the capital of France?",
    target="Paris",
    output_span=(0, 0),
    hops=1,
)

print(trace.topk_inputs(10))
trace.to_html("trace.html")
trace.to_json("trace.json")
```

## CLI Quickstart

```bash
flashtrace trace \
  --model Qwen/Qwen3-8B \
  --prompt prompt.txt \
  --target target.txt \
  --output-span 0:0 \
  --hops 1 \
  --html trace.html \
  --json trace.json
```

## Token Spans

`output_span` and `reasoning_span` use inclusive generation-token indices. Inspect `trace.generation_tokens` after an initial run to choose spans for a target answer or reasoning segment.

## Supported Models

The package targets Llama/Qwen-style decoder-only Hugging Face causal LMs with standard Q/K/V/O projections, RMSNorm or LayerNorm, and RoPE metadata. Qwen2, Qwen3, and Llama are the first validated model families.

## Repository Map

- `flashtrace/`: reusable package
- `examples/`: public examples
- `tests/`: CPU smoke tests
- `exp/`: paper experiments and artifacts

## Citation

```bibtex
@misc{pan2026flashtrace,
  title={Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs},
  author={Pan, Wenbo and Liu, Zhichao and Wang, Xianlong and Yu, Haining and Jia, Xiaohua},
  year={2026},
  eprint={2602.01914},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
```

- [ ] **Step 3: Add MIT license**

Create `LICENSE`:

```text
MIT License

Copyright (c) 2026 Wenbo Pan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Update generated artifact ignore rules**

Append to `.gitignore`:

```gitignore

# FlashTrace generated artifacts
trace.json
trace.html
*.trace.json
*.trace.html
exp/**/output/
exp/**/out/
exp/**/out-*/
*.npz
```

- [ ] **Step 5: Remove the template artifact**

Run:

```bash
git rm model_generation.py
```

- [ ] **Step 6: Verify quickstart help**

Run:

```bash
uv run python examples/quickstart.py --help
```

Expected: help text includes `FlashTrace quickstart example`.

- [ ] **Step 7: Commit**

Run:

```bash
git add README.md LICENSE examples/quickstart.py .gitignore
git commit -m "docs: add public quickstart and release hygiene"
```

## Task 8: Final Verification And Package Audit

**Files:**
- Modify: any files needed to fix verification failures.

- [ ] **Step 1: Run the full CPU test suite**

Run:

```bash
uv run pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify editable install import**

Run:

```bash
uv run python -c "import flashtrace; print(flashtrace.FlashTrace.__name__)"
```

Expected: prints `FlashTrace`.

- [ ] **Step 3: Verify CLI help**

Run:

```bash
uv run flashtrace --help
uv run flashtrace trace --help
```

Expected: both commands print help text.

- [ ] **Step 4: Verify root compatibility imports**

Run:

```bash
uv run python -c "import ifr_core, llm_attr, ft_ifr_improve; print(ifr_core.compute_multi_hop_ifr.__name__)"
```

Expected: prints `compute_multi_hop_ifr`.

- [ ] **Step 5: Inspect package file list**

Run:

```bash
git status --short
find flashtrace -maxdepth 3 -type f | sort
```

Expected: package files match the design spec and only intended changes appear.

- [ ] **Step 6: Commit final fixes**

Run after any verification fixes:

```bash
git add .
git commit -m "test: verify public package smoke tests"
```

If verification passes with a clean tree after prior commits, record the passing commands in the final implementation response.

## Self-Review Checklist

- Spec coverage: package layout, public API, result export, CLI, visualization, packaging, compatibility, tests, README, and release hygiene each have at least one task.
- Type consistency: `FlashTrace.trace`, `TraceResult`, `TokenScore`, `load_model_and_tokenizer`, and CLI span parsing use the same names across tests and implementation steps.
- Test path: every implementation task starts with a failing test or a verification command, then ends with a passing command and commit.
