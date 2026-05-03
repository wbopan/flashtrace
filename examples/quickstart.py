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
