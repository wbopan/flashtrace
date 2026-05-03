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
