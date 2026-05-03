# Qwen Token Overlay E2E Design

## Goal

Build a Qwen-backed live demo that generates its own reasoning and final answer, maps the generated text through the real Qwen tokenizer, renders a word-level attribution overlay, and lets the user select visible boxes that map back to exact tokenizer token spans.

## Success Criteria

- The demo starts from a prompt or dataset sample and calls Qwen to produce the response.
- The response includes intermediate reasoning and a final answer segment.
- The overlay uses real tokenizer offsets from the selected Qwen tokenizer.
- Visible word boxes are selectable and map to generation-token spans.
- Control tokens such as `<bos>`, `<eos>`, `<|im_start|>`, and `<|im_end|>` remain inspectable.
- FlashTrace receives the selected `reasoning_span` and `output_span` as inclusive generation-token spans.
- The Paris smoke scenario still ranks the prompt word `Paris` highest after tracing the selected final answer.

## Product Shape

The demo has three phases.

1. `Generate with Qwen`: runs the model with deterministic generation settings and stores the raw generated text, generated token ids, and tokenizer offsets.
2. `Inspect / Select`: shows the generated reasoning and final answer as selectable word boxes, with a raw token inspector for special tokens and tokenizer boundaries.
3. `Trace selected answer`: runs FlashTrace with the selected answer span and the parsed or selected reasoning span, then overlays attribution scores on prompt-side word boxes.

This shape makes long model loading, generation, span inspection, and attribution progress visible as separate operations.

## Core Coordinate System

The tokenizer is the single coordinate system.

All display objects derive from `AutoTokenizer(..., return_offsets_mapping=True)`.
The UI groups tokens into word boxes for readability. Each word box stores its underlying token indices. Selections always resolve to token indices before calling FlashTrace.

For generated text, offsets are computed over the exact `generation_text` passed to FlashTrace as `target`. This keeps parser spans, visible selections, and attribution spans aligned.

For prompt text, the display uses the same prompt string and chat-template setting as the attribution path. When chat templates are enabled, the user prompt and template/control tokens are represented separately so the prompt heatmap stays aligned with FlashTrace's prompt-token projection.

## Data Model

`demo/live/token_overlay.py` will own the overlay data structures.

```python
from dataclasses import dataclass
from typing import Literal

Section = Literal["prompt", "thinking", "answer", "other"]
TokenKind = Literal["content", "whitespace", "special", "template", "control"]

@dataclass(frozen=True)
class TokenRecord:
    section: Section
    token_index: int
    token_id: int | None
    token_text: str
    char_start: int
    char_end: int
    kind: TokenKind
    selectable: bool
    role: str

@dataclass(frozen=True)
class WordBox:
    box_id: str
    section: Section
    text: str
    char_start: int
    char_end: int
    token_indices: tuple[int, ...]
    kind: TokenKind
    selectable: bool
    score: float | None = None

@dataclass(frozen=True)
class GenerationSections:
    generation_text: str
    thinking_char_span: tuple[int, int] | None
    answer_char_span: tuple[int, int]
    thinking_token_span: tuple[int, int] | None
    answer_token_span: tuple[int, int]
    parser: str
```

`TokenRecord.token_index` is local to its section. For generated text, it is the generation-token index expected by `FlashTrace.trace(... output_span=..., reasoning_span=...)`.

## Generation Flow

The Qwen path loads model and tokenizer through the existing demo model cache. The generator uses deterministic settings for repeatable demos:

```python
generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}
```

The prompt instructs the model to produce reasoning and a final answer with explicit markers. The preferred format is:

```text
<think>
...
</think>
<answer>
...
</answer>
```

The demo stores both:

- `display_generation_text`: decoded with regular display cleanup for reading.
- `trace_generation_text`: decoded with stable tokenizer alignment for span mapping and FlashTrace target input.

The first implementation can use one text field when decoding stays stable for the chosen Qwen model. A later refinement can add a cached-generation FlashTrace path that consumes captured token ids directly.

## Span Detection

Span detection runs in this order:

1. `<think>...</think>` plus `<answer>...</answer>` parser.
2. `\boxed{...}` parser used by the paper experiments.
3. Final paragraph parser where earlier text is reasoning and the last paragraph is the final answer.

Detected character spans are converted to token spans by overlap against tokenizer offsets:

```python
def char_span_to_token_span(offsets, start_char, end_char):
    indices = [
        i for i, (start, end) in enumerate(offsets)
        if start < end_char and end > start_char
    ]
    if not indices:
        raise ValueError("No tokenizer tokens overlap the selected character span")
    return min(indices), max(indices)
```

The overlap rule matches the existing experiment helper in `exp/exp2/dataset_utils.py`.

## Word Grouping

Word boxes are a display layer over token records.

Grouping rules:

- Consecutive content tokens with adjacent non-whitespace character spans form one word box when they belong to the same visible word.
- Whitespace tokens become separators and remain visible in the raw token inspector.
- Punctuation can join the neighboring word for compact reading and remains independently traceable through the raw token inspector.
- Special/control/template tokens become compact chips in the raw lane.

Box score aggregation uses max absolute token score for color intensity. Tooltips show max, mean, sum, token ids, token texts, and token indices.

## Special Token Handling

Special and control tokens stay in the data model and raw token inspector. They are collapsed by default in the visible reading overlay.

Rules:

- `<bos>` and chat-template start tokens get `kind="control"` or `kind="template"` and `selectable=False`.
- `<eos>` gets `kind="control"` and `selectable=False`.
- `<think>`, `</think>`, `<answer>`, and `</answer>` get `kind="control"` when they are tokenizer-visible markers.
- A click on a neighboring visible word selects content tokens only.
- Manual range selection snaps to selectable content tokens and preserves the hidden control tokens in the inspector.

This keeps the human-facing selection clean while preserving exact tokenizer evidence.

## Gradio UI

The live page will use one screen with four areas:

- Left: prompt input, model selector, sample selector, generation settings, Generate button.
- Center: generated response overlay with Thinking and Final Answer bands.
- Right: raw token inspector with token index, token id, token text, offsets, kind, and role.
- Bottom: prompt attribution overlay, top-k table, per-hop summary, and JSON/HTML export links.

Selection behavior:

- Clicking a word box in the Final Answer band sets `output_span`.
- Clicking a word box in the Thinking band previews a reasoning-token range.
- A `Use auto reasoning span` toggle uses the parsed full reasoning span.
- A `Show control tokens` toggle expands the raw token lane.

## FlashTrace Integration

The trace call uses the existing public facade:

```python
trace = tracer.trace(
    prompt=prompt,
    target=sections.generation_text,
    output_span=sections.answer_token_span,
    reasoning_span=sections.thinking_token_span,
    hops=1,
    method="flashtrace",
)
```

`target` comes from Qwen generation captured in the first phase. The user can inspect and adjust spans before the call. The trace result scores align with `trace.prompt_tokens`, and the prompt overlay groups those tokens into word boxes using the same tokenizer-offset strategy.

## Error Handling

Model loading errors show a short actionable message with the model id and the failing phase.

Generation errors preserve the prompt and model settings in the UI state.

Parser failures show the raw generation and offer manual final-answer selection from word boxes.

Tokenizer offset failures stop the trace phase and show the tokenizer name, text length, and section being mapped.

Trace errors keep the selected spans visible so the user can retry with adjusted spans.

Long Qwen downloads are surfaced as a model-loading phase. The local demo keeps the smoke model as the default and offers Qwen as an explicit model choice.

## Testing

Unit tests cover:

- special token classification,
- char span to token span conversion,
- word grouping from real tokenizer offsets,
- parser behavior for think/answer, boxed answer, and final paragraph formats,
- output span and reasoning span serialization into the Gradio state.

Integration tests cover:

- real Qwen tokenizer offset mapping for a short prompt and generated response,
- mocked Qwen generation that includes thinking and final answer markers,
- FlashTrace call construction with inclusive spans.

Browser e2e covers:

- open `http://127.0.0.1:7860/`,
- select Qwen or smoke model,
- click Generate,
- verify thinking and final-answer bands render,
- click a final-answer word box,
- click Trace selected answer,
- verify the prompt word `Paris` receives the highest prompt-side score in the Paris scenario.

## Implementation Boundary

This feature changes the live demo and small helper modules. It reuses the public `FlashTrace.trace` facade and existing model loading path. Core attribution math remains unchanged.

The first version prioritizes faithful tokenizer alignment, visible selections, and a reliable Qwen e2e path. A later version can add cached-token-id tracing to remove the final target re-tokenization step.
