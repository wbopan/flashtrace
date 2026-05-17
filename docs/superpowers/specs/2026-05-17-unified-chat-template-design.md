# Unify Chat-Template Handling — Design

Date: 2026-05-17
Status: Approved (design); pending spec review

## Goal

Make the demo apply each model's **official chat template** consistently in all
three places that currently disagree: generation, trace, and document display.
Remove the per-request toggle — the chat template is **always** applied.

## Current state (the inconsistency)

1. **Display** (`service.py::_prompt_text_for_document`) — applies the official
   `tokenizer.apply_chat_template([{role: user, content: prompt}], ...)`. Clean.
2. **Generation** (`qwen_generation.py::generate_with_qwen`) — `tokenizer(prompt)`,
   **no chat template at all**.
3. **Trace** (`flashtrace/attribution.py::format_prompt`) — applies
   `apply_chat_template`, but first wraps the prompt in the private
   `DEFAULT_PROMPT_TEMPLATE` (`"Context:{context}\n\n\nQuery: {query}"`), passes
   `enable_thinking=False`, and prepends a space (`" " + prompt`).

So the string the model generates from, the string FlashTrace attributes, and
the tokens the user sees are three different things.

## The unified contract

For a user prompt `P`, all three paths produce the model input from the single
canonical call:

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": P}],
    add_generation_prompt=True,
    tokenize=...,   # False for display, True/ids for generation
)
```

- No `DEFAULT_PROMPT_TEMPLATE` wrapper.
- No `enable_thinking` argument — the model's official chat-template default is
  used (for Qwen3 this leaves thinking enabled; the demo's `detect_sections`
  already handles `<think>` blocks).
- No `" " + prompt` leading space.
- No system message — a single user turn, everywhere.

## Changes

### 1. Core library — `flashtrace/attribution.py`

- `format_prompt`: when `use_chat_template` is true, set the user message
  content to the prompt **directly** (drop `DEFAULT_PROMPT_TEMPLATE`, drop
  `enable_thinking=False`).
- `response` / `target_response`: drop the `" " + prompt` leading space; set
  `self.user_prompt = prompt`.
- `user_prompt_indices` (the logic that locates the user-prompt token
  subsequence inside the formatted prompt for prompt-side attribution) must
  stay correct without the leading-space hack. The current matcher keys on the
  first user-prompt token id; make it robust to the chat-template boundary
  (e.g. search for the full `user_prompt_ids` subsequence rather than only the
  first-token match, and fail loudly if not found).
- `DEFAULT_PROMPT_TEMPLATE` **stays in `shared_utils.py`** — `exp/` scripts and
  `llm_attr_eval.py` import it independently; only its use inside
  `format_prompt` is removed.
- The `use_chat_template` constructor flag stays (cli / exp / other callers
  rely on it). Only its *behavior when true* changes (clean template, no
  wrapper).

### 2. Demo generation — `demo/live/qwen_generation.py`

`generate_with_qwen` builds the model input through the chat template:
`apply_chat_template([{role: user, content: prompt}], add_generation_prompt=True,
return_tensors="pt")`, then `model.generate(...)`. `output.token_ids` /
`output.text` remain the real generated tokens (the tokenization contract for
the generation region is unchanged — the chat template only affects the prompt
side).

### 3. Demo service / API — `demo/live/service.py`, `server.py`

- Remove the `chat_template` field from `TokenizeRequest`, `GenerateRequest`,
  `TraceRequest`.
- Remove the `use_chat_template` parameter from the service functions
  (`run_prompt_document_phase`, `run_generate_document_phase`,
  `run_trace_document_phase`); the chat template is always applied.
- `run_trace_document_phase` constructs `FlashTrace(..., use_chat_template=True)`
  unconditionally.
- A single demo-side helper formats the prompt for both generation and display
  so the two are byte-identical:
  `format_chat_prompt(prompt, tokenizer) -> str` (or returns ids for
  generation). FlashTrace's `format_prompt` independently produces the
  equivalent canonical string.

### 4. Frontend — `demo/live/static/`

- Remove the `Show chat template` checkbox from `index.html`.
- `app.js`: drop the checkbox element, its `change` handler, and the
  `chat_template` field from `basePayload()`. Auto-tokenize still fires on load
  and on prompt change.

## Alignment outcome

- The prompt-phase document shows the chat-template-formatted token sequence
  (the official template's tokens, e.g. `<|im_start|>`), built from
  `build_token_records` over the formatted string.
- The model generates from that same formatted string.
- FlashTrace traces over that same formatted string; the traced-phase document
  renders FlashTrace's own tokens, now produced from the identical template.
- All three are byte-identical for a given prompt.

## Error handling

- A tokenizer without a `chat_template` (`apply_chat_template` raises): surface
  a clear `ValueError` ("model has no chat template") through the existing
  `400` path. The default model (`Qwen/Qwen3-0.6B`) has one.

## Testing

- `format_prompt` unit test: `use_chat_template=True` output equals
  `apply_chat_template([{role: user, content: prompt}], add_generation_prompt=True)`
  with no `Context:`/`Query:` wrapper and no leading space.
- `user_prompt_indices` test: the located subsequence decodes back to the
  prompt for a chat-templated input (tiny Qwen2 model).
- `generate_with_qwen` test: the model input is the chat-templated prompt
  (assert the formatted-prompt ids are the generation input prefix).
- Demo service / API tests: drop `chat_template` from request payloads; assert
  prompt-phase tokens include the template markers.
- Frontend: the Playwright smoke test no longer references the checkbox.
- Full suite stays green.

## Risks

- Removing the `" " + prompt` leading space can break `user_prompt_indices`
  subsequence matching at the chat-template boundary. Mitigation: full
  subsequence search + loud failure; covered by a dedicated test. This is the
  main implementation risk.
- `format_prompt` is core-library and on the demo's trace hot path
  (`improved.py` calls it). `exp/` scripts pre-format with
  `DEFAULT_PROMPT_TEMPLATE` and pass `use_chat_template=False`, so they are not
  expected to change — but this is a library behavior change for any caller
  that used `use_chat_template=True`.
