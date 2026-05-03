<p align="center">
  <img src="https://raw.githubusercontent.com/wbopan/flashtrace/master/docs/assets/flashtrace-logo.png" alt="FlashTrace logo" width="160">
</p>

<h1 align="center">FlashTrace</h1>

<p align="center">
  <em>Fast token attribution for reasoning language models.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/flashtrace/"><img alt="PyPI" src="https://img.shields.io/pypi/v/flashtrace.svg?style=flat-square&logo=pypi&logoColor=white&label=PyPI"></a>
  <a href="https://pypi.org/project/flashtrace/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/flashtrace.svg?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://github.com/wbopan/flashtrace/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2602.01914"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.01914-B31B1B.svg?style=flat-square&logo=arxiv&logoColor=white"></a>
</p>

FlashTrace traces generated answers back to the prompt tokens that shaped them. Use it from Python or the command line, export JSON traces, and render standalone HTML heatmaps for inspection and sharing.

<p align="center">
  <a href="https://arxiv.org/abs/2602.01914">📄 Paper</a>
  &nbsp;·&nbsp;
  <a href="#quickstart">🚀 Quickstart</a>
  &nbsp;·&nbsp;
  <a href="#command-line">💻 CLI</a>
  &nbsp;·&nbsp;
  <a href="#citation">📝 Citation</a>
</p>

## Why FlashTrace

Reasoning models produce long generated chains, final answers, and intermediate spans that deserve targeted inspection. FlashTrace gives researchers a package-first workflow for tracing a selected generated span back to its supporting prompt tokens.

You get:

- top-k prompt tokens ranked by attribution score
- JSON traces for downstream analysis
- standalone HTML token heatmaps
- optional per-hop attribution panels
- inclusive generation-token span controls for answer and reasoning segments

## Install

From PyPI:

```bash
pip install flashtrace
```

From a local checkout:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

FlashTrace uses PyTorch, Transformers, Accelerate, NumPy, and tqdm. A CUDA-capable GPU is recommended for public-scale Hugging Face models.

## Quickstart

```python
from flashtrace import FlashTrace, load_model_and_tokenizer

prompt = """Context: Paris is the capital of France.
Question: What is the capital of France?"""
target = "Paris"

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-8B", device_map="auto")
tracer = FlashTrace(model, tokenizer, chunk_tokens=128, sink_chunk_tokens=32)

trace = tracer.trace(
    prompt=prompt,
    target=target,
    output_span=(0, 0),
    hops=1,
)

print(trace.topk_inputs(10))
trace.to_json("trace.json")
trace.to_html("trace.html")
```

`trace.topk_inputs(10)` returns `TokenScore` objects aligned to prompt-token indices:

```text
rank  index  token      score
1     2      Paris      0.184
2     7      capital    0.131
3     10     France     0.119
```

`trace.html` is a standalone heatmap that highlights prompt tokens by final attribution score and includes trace metadata for the selected generated span.

`FlashTrace(..., use_chat_template=True)` formats prompts with the tokenizer chat template for chat-tuned models.

## Command Line

Create prompt and target files:

```bash
printf "Context: Paris is the capital of France.\nQuestion: What is the capital of France?\n" > prompt.txt
printf "Paris" > target.txt
```

Run a trace:

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

The command prints a compact top-k table and writes the requested artifacts.

Useful flags:

- `--model`: Hugging Face model id or local model path
- `--prompt`: UTF-8 prompt text file
- `--target`: UTF-8 target text file
- `--output-span`: inclusive `START:END` indices over generated tokens
- `--reasoning-span`: inclusive `START:END` indices for a reasoning segment
- `--method`: `flashtrace`, `ifr-span`, or `ifr-matrix`
- `--recompute-attention`: lower-memory attention recomputation path
- `--use-chat-template`: format prompts with the tokenizer chat template
- `--device-map`: Transformers device map, default `auto`
- `--dtype`: `auto`, `float16`, `bfloat16`, or `float32`

## Token Spans

`output_span` and `reasoning_span` use inclusive generation-token indices. The first generated token has index `0`.

Use an initial trace to inspect tokenization:

```python
for index, token in enumerate(trace.generation_tokens):
    print(index, repr(token))
```

Then choose spans:

```python
trace = tracer.trace(
    prompt=prompt,
    target=target,
    reasoning_span=(0, 79),
    output_span=(80, 85),
    hops=1,
)
```

Scores are aligned to `trace.prompt_tokens`. `trace.per_hop_scores` stores the same prompt-token alignment for each hop.

## Interpreting Results

High-scoring prompt tokens are the tokens FlashTrace attributes most strongly to the selected generated span. For answer inspection, use `output_span` around the final answer tokens. For chain-of-thought or reasoning inspection, use `reasoning_span` around the generated reasoning segment.

Recommended workflow:

1. Run a trace with your prompt and target.
2. Inspect `trace.generation_tokens`.
3. Select the answer or reasoning span.
4. Export `trace.html`.
5. Compare top-k tokens with the source prompt and any expected evidence.

## Supported Models

FlashTrace targets Llama/Qwen-style decoder-only Hugging Face causal LMs with:

- `model.layers`
- Q/K/V/O attention projections
- RMSNorm or LayerNorm
- RoPE metadata

Validated model families for the first public release:

- Qwen2
- Qwen3
- Llama

## Python API

The public package exports:

```python
from flashtrace import FlashTrace, TraceResult, load_model_and_tokenizer
```

`FlashTrace.trace(...)` accepts:

- `prompt: str`
- `target: str | None`
- `output_span: tuple[int, int] | None`
- `reasoning_span: tuple[int, int] | None`
- `hops: int`
- `method: "flashtrace" | "ifr-span" | "ifr-matrix"`
- `renorm_threshold: float | None`

`TraceResult` includes:

- `prompt_tokens`
- `generation_tokens`
- `scores`
- `per_hop_scores`
- `thinking_ratios`
- `output_span`
- `reasoning_span`
- `method`
- `metadata`

Export helpers:

```python
trace.topk_inputs(20)
trace.to_dict()
trace.to_json("trace.json")
trace.to_html("trace.html")
```

## Examples

```bash
python examples/quickstart.py --help
python examples/quickstart.py \
  --model Qwen/Qwen3-8B \
  --prompt "Context: Paris is the capital of France. Question: What is the capital of France?" \
  --target "Paris" \
  --output-span 0:0 \
  --html trace.html
```

Heavy model examples are intended for GPU environments. CPU smoke tests use tiny randomly initialized models.

## Repository Map

- `flashtrace/`: reusable Python package
- `examples/`: public quickstarts
- `tests/`: CPU smoke tests
- `exp/`: paper experiments and research artifacts
- `docs/superpowers/`: design and implementation planning documents

## Research Experiments

The `exp/` directory contains the paper-era experiment runners, case studies, and saved artifacts. The public package API lives in `flashtrace/`; experiment scripts keep compatibility imports during the package migration.

## Troubleshooting

**CUDA memory**

Use smaller models, lower precision, `device_map="auto"`, shorter prompts, or `--recompute-attention`.

**Span selection**

Print `trace.generation_tokens` and select inclusive generated-token indices. Tokenization can split visible words into multiple model tokens.

**Deterministic generation**

Pass a `target` file for attribution against a known output. Leave `--target` out when you want the CLI to generate with deterministic defaults.

**Tokenizer alignment**

Inspect `trace.prompt_tokens` and `trace.generation_tokens` when scores appear shifted from visible text. Attribution scores follow tokenizer-level alignment.

**HTML export**

`trace.to_html("trace.html")` writes a standalone file that can be opened locally or shared as an artifact.

## Paper

FlashTrace implements the method described in [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](https://arxiv.org/abs/2602.01914).

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
