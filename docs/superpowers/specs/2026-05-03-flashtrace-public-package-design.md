# FlashTrace Public Package Design

## Goal

Turn the current FlashTrace research repository into an installable, documented Python package that researchers can use from Python or the command line to trace LLM outputs, export JSON traces, and render HTML token heatmaps.

## Release Scope

This first public release ships four user-facing capabilities:

- A stable Python API centered on `FlashTrace`.
- A `flashtrace trace` CLI for prompt/target files and Hugging Face model ids or local model paths.
- A `TraceResult` object with top-k, JSON, and HTML export helpers.
- A README quickstart that demonstrates Python, CLI, and heatmap workflows.

Paper experiment runners and saved experiment artifacts remain in `exp/` as research assets. Their full reproducibility cleanup belongs to a later phase.

## Repository Shape

The reusable package lives under `flashtrace/`, examples under `examples/`, tests under `tests/`, and paper experiments under `exp/`.

```text
flashtrace/
  __init__.py
  tracer.py
  result.py
  core.py
  model_io.py
  viz.py
  cli.py
  baselines/
    __init__.py
    attnlrp.py
examples/
  quickstart.py
tests/
  test_core_recompute.py
  test_tracer.py
  test_result.py
  test_cli.py
exp/
  exp1/
  exp2/
  case_study/
```

Existing root modules are migrated gradually. During migration, compatibility wrappers remain at the root for experiment scripts that still import `llm_attr`, `ifr_core`, or `ft_ifr_improve`.

## Core Implementation Mapping

`flashtrace.core` contains the IFR tensor implementation from `ifr_core.py`:

- `extract_model_metadata`
- `build_weight_pack`
- `attach_hooks`
- `recompute_layer_attention`
- `compute_ifr_sentence_aggregate`
- `compute_multi_hop_ifr`
- `compute_ifr_for_all_positions`

`flashtrace.tracer` wraps the current high-level attribution classes:

- Default `method="flashtrace"` uses the current `LLMIFRAttributionBoth.calculate_ifr_multi_hop_both` behavior.
- `method="ifr-span"` uses `LLMIFRAttribution.calculate_ifr_span`.
- `method="ifr-matrix"` uses `LLMIFRAttribution.calculate_ifr_for_all_positions_output_only`.

`flashtrace.baselines.attnlrp` contains the AttnLRP patching and recursive baseline code from `lrp_rules.py`, `lrp_patches.py`, and `LLMLRPAttribution`.

## Public Python API

The package exports `FlashTrace`, `TraceResult`, and `load_model_and_tokenizer`.

```python
from flashtrace import FlashTrace, load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-8B", device_map="auto")
tracer = FlashTrace(model, tokenizer, chunk_tokens=128, sink_chunk_tokens=32)

trace = tracer.trace(
    prompt=prompt,
    target=target,
    output_span=(80, 85),
    reasoning_span=(0, 79),
    hops=1,
)

print(trace.topk_inputs(20))
trace.to_json("trace.json")
trace.to_html("trace.html")
```

`FlashTrace.trace(...)` accepts:

- `prompt: str`
- `target: str | None`
- `output_span: tuple[int, int] | None`
- `reasoning_span: tuple[int, int] | None`
- `hops: int`
- `method: Literal["flashtrace", "ifr-span", "ifr-matrix"]`
- `renorm_threshold: float | None`

Generation-token spans are inclusive and use the tokenizer alignment already produced by the attribution path. The README explains this convention and shows how to inspect `trace.generation_tokens`.

## TraceResult

`TraceResult` is a small dataclass that hides the older `LLMAttributionResult` shape from public users.

Fields:

- `prompt_tokens: list[str]`
- `generation_tokens: list[str]`
- `scores: list[float]`
- `per_hop_scores: list[list[float]]`
- `thinking_ratios: list[float]`
- `output_span: tuple[int, int] | None`
- `reasoning_span: tuple[int, int] | None`
- `method: str`
- `metadata: dict[str, Any]`

Methods:

- `topk_inputs(k: int = 20) -> list[TokenScore]`
- `to_dict() -> dict[str, Any]`
- `to_json(path: str | Path) -> None`
- `to_html(path: str | Path) -> None`

`TokenScore` contains `index`, `token`, and `score`. Scores are aligned to `prompt_tokens`.

## CLI

The package exposes one console script:

```bash
flashtrace trace \
  --model Qwen/Qwen3-8B \
  --prompt prompt.txt \
  --target target.txt \
  --output-span 80:85 \
  --reasoning-span 0:79 \
  --hops 1 \
  --html trace.html \
  --json trace.json
```

CLI behavior:

- `--model` accepts a Hugging Face id or local path.
- `--prompt` and `--target` read UTF-8 text files.
- `--target` is optional; the model generates with deterministic defaults when this flag is absent.
- `--output-span` and `--reasoning-span` parse inclusive `START:END` generation-token spans.
- `--method` defaults to `flashtrace`.
- `--recompute-attention` enables lower-memory attention recomputation.
- `--device-map` defaults to `auto`.
- `--dtype` accepts `auto`, `float16`, `bfloat16`, or `float32`.

The command prints a compact top-k table to stdout and writes requested artifacts.

## Visualization

`flashtrace.viz` adapts the token heatmap renderer from `exp/case_study/viz.py`.

The public heatmap focuses on:

- prompt tokens colored by final attribution score,
- optional per-hop panels,
- output and reasoning span summary,
- model/method metadata.

The renderer returns a standalone HTML string and writes standalone HTML files through `TraceResult.to_html`.

## Packaging

`pyproject.toml` becomes package metadata for `flashtrace`:

- `name = "flashtrace"`
- realistic `requires-python` support for current PyTorch and Transformers use,
- console script `flashtrace = "flashtrace.cli:main"`,
- core dependencies: `torch`, `transformers`, `accelerate`, `numpy`, `tqdm`,
- optional extras: `viz`, `eval`, `dev`, `baselines`.

The root README includes:

- project tagline,
- paper link and citation,
- install instructions,
- Python quickstart,
- CLI quickstart,
- supported model family notes,
- output interpretation,
- experiment directory map,
- troubleshooting for GPU memory and tokenizer spans.

## Compatibility

The release supports Llama/Qwen-style decoder-only Hugging Face causal LMs with `model.layers`, Q/K/V/O projections, RMSNorm/LayerNorm, and RoPE metadata. The README names Qwen2, Qwen3, and Llama as validated families.

Existing experiment scripts continue to run through temporary root-level compatibility modules while package imports are introduced. A later cleanup can remove the compatibility layer after `exp/` imports are migrated.

## Testing

Tests use a tiny randomly initialized Qwen2 model on CPU, following the existing `test_recompute.py` approach.

Required coverage:

- stored-attention and recomputed-attention paths return close values on the tiny model,
- `FlashTrace.trace(...)` returns a `TraceResult`,
- `TraceResult.topk_inputs(...)` sorts and truncates correctly,
- `TraceResult.to_dict()` is JSON serializable,
- `TraceResult.to_html()` writes standalone HTML containing token spans,
- `flashtrace trace --help` exits successfully.

Heavy GPU model tests remain manual examples.

## Release Hygiene

The release cleanup updates `.gitignore` to cover generated traces, experiment outputs, checkpoints, caches, and HTML/JSON artifacts created by examples.

Tracked historical experiment outputs stay untouched during the first package migration. A later artifact cleanup can move them to release assets or remove them with a dedicated confirmation step.

`model_generation.py` is a template artifact and is removed or moved outside the package path during implementation.

## Success Criteria

The release work is complete when:

- `pip install -e .` exposes `flashtrace`,
- `python examples/quickstart.py --help` works,
- `flashtrace trace --help` works,
- package smoke tests pass on CPU,
- README quickstart matches the implemented API,
- existing experiment entrypoints either run with compatibility imports or document their package-era invocation.
