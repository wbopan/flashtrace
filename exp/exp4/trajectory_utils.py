"""Data model and rendering helpers for the multi-turn Aider benchmark.

The on-disk trajectory format is model agnostic: it stores chat ``messages``
and leaves chat-template rendering to the attribution target tokenizer.  This
lets the same sampled trajectory be evaluated by different Qwen3 checkpoints
without baking one checkpoint's special-token layout into the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class TrajectorySegment:
    """One message-content span inside the rendered attribution prompt."""

    message_index: int
    role: str
    kind: str
    turn: int
    char_span: Tuple[int, int]


@dataclass(frozen=True)
class AiderExample:
    """An attribution-ready legacy Aider sample or multi-turn trajectory."""

    prompt: str
    target: str
    metadata: Dict[str, Any]
    messages: Tuple[Dict[str, str], ...] = ()
    segments: Tuple[TrajectorySegment, ...] = ()
    sink_span: Optional[Tuple[int, int]] = None
    thinking_span: Optional[Tuple[int, int]] = None

    @property
    def is_multiturn(self) -> bool:
        assistant_turns = sum(message.get("role") == "assistant" for message in self.messages)
        return assistant_turns >= 2


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def normalize_messages(
    raw_messages: Any,
    *,
    allow_empty_content: bool = False,
) -> List[Dict[str, str]]:
    """Validate and normalize a stored trajectory message list."""

    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Trajectory row must contain a non-empty messages list.")

    messages: List[Dict[str, str]] = []
    for idx, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ValueError(f"messages[{idx}] must be an object.")
        role = str(raw.get("role") or "").strip().lower()
        if role not in ALLOWED_ROLES:
            raise ValueError(f"messages[{idx}].role must be one of {sorted(ALLOWED_ROLES)}; got {role!r}.")
        content = str(raw.get("content") or "")
        if not content.strip() and not allow_empty_content:
            raise ValueError(f"messages[{idx}].content must be non-empty.")
        kind = str(raw.get("kind") or ("task" if role == "user" else "response"))
        messages.append({"role": role, "content": content, "kind": kind})

    if messages[-1]["role"] != "assistant":
        raise ValueError("The final trajectory message must be an assistant response used as the attribution target.")

    non_system_roles = [message["role"] for message in messages if message["role"] != "system"]
    if not non_system_roles or non_system_roles[0] != "user":
        raise ValueError("The first non-system trajectory message must be a user task.")
    for prev, current in zip(non_system_roles, non_system_roles[1:]):
        if prev == current:
            raise ValueError("Non-system trajectory messages must alternate user and assistant roles.")
    return messages


def _apply_chat_template(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    template_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        rendered = tokenizer.apply_chat_template(
            template_messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(template_messages, **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Tokenizer chat template returned an empty/non-string prompt.")
    return rendered


def _render_plain(messages: Sequence[Dict[str, str]]) -> str:
    """Readable fallback used only when no target tokenizer is supplied."""

    pieces: List[str] = []
    for message in messages:
        pieces.append(f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n")
    pieces.append("<assistant>\n")
    return "".join(pieces)


def _locate_message_segments(
    rendered_prompt: str,
    messages: Sequence[Dict[str, str]],
    *,
    allow_missing_assistant: bool = False,
) -> Tuple[TrajectorySegment, ...]:
    """Locate message contents sequentially in a rendered chat prompt.

    Qwen chat templates preserve message content verbatim.  Sequential search
    disambiguates repeated snippets while producing tokenizer-independent char
    spans that can later be mapped with offset mappings.
    """

    segments: List[TrajectorySegment] = []
    cursor = 0
    role_turns: Dict[str, int] = {"system": 0, "user": 0, "assistant": 0}
    for message_index, message in enumerate(messages):
        content = message["content"]
        start = rendered_prompt.find(content, cursor)
        if start < 0:
            if not (allow_missing_assistant and message["role"] == "assistant"):
                raise ValueError(
                    "Could not locate a trajectory message verbatim in the target tokenizer's "
                    f"rendered chat prompt (message_index={message_index}, role={message['role']})."
                )
            start, end = -1, -1
        else:
            end = start + len(content)
        role = message["role"]
        role_turns[role] += 1
        segments.append(
            TrajectorySegment(
                message_index=message_index,
                role=role,
                kind=message.get("kind", "response"),
                turn=role_turns[role],
                char_span=(start, end),
            )
        )
        if end >= 0:
            cursor = end
    return tuple(segments)


def _char_span_to_token_span(
    tokenizer: Any,
    text: str,
    span: Sequence[int] | None,
) -> Optional[Tuple[int, int]]:
    if tokenizer is None or span is None or len(span) != 2:
        return None
    start_char, end_char = int(span[0]), int(span[1])
    if end_char <= start_char:
        return None
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    indices = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if int(start) < end_char and int(end) > start_char
    ]
    if not indices:
        return None
    return min(indices), max(indices)


def render_trajectory(
    messages: Sequence[Dict[str, str]],
    *,
    tokenizer: Optional[Any] = None,
) -> Tuple[str, str, Tuple[TrajectorySegment, ...]]:
    """Render all history as prompt and the final assistant turn as target."""

    normalized = normalize_messages(list(messages))
    history = normalized[:-1]
    if not history:
        raise ValueError("Trajectory must contain history before the final assistant target.")

    if tokenizer is None:
        prompt = _render_plain(history)
    else:
        prompt = _apply_chat_template(tokenizer, history)
    target = normalized[-1]["content"]
    segments = _locate_message_segments(prompt, history)
    return prompt, target, segments


def load_aider(path: Path, *, tokenizer: Optional[Any] = None) -> List[AiderExample]:
    """Load legacy exp4 rows and schema-v2 multi-turn trajectories.

    Legacy rows use ``input``/``output`` exactly as before.  New rows contain a
    ``messages`` list and are rendered with the attribution target tokenizer's
    official chat template.
    """

    examples: List[AiderExample] = []
    for row_index, row in enumerate(_read_jsonl(path)):
        if (
            row.get("benchmark") == "aider_polyglot"
            and int(row.get("schema_version") or 0) >= 3
            and row.get("turns")
        ):
            if not row.get("valid_final_action"):
                continue
            final = row["turns"][-1]
            prompt = str(final.get("rendered_prompt") or "")
            target = str(final.get("completion") or "")
            if not prompt or not target:
                raise ValueError(
                    f"Polyglot row {row_index} has an empty final prompt or target."
                )
            history = normalize_messages(
                list(row.get("messages") or [])
                + [
                    {
                        "role": "assistant",
                        "content": target,
                        "kind": final.get("kind", "final_edit"),
                    }
                ],
                allow_empty_content=True,
            )
            segments = _locate_message_segments(
                prompt,
                history[:-1],
                allow_missing_assistant=True,
            )
            sink_span = _char_span_to_token_span(
                tokenizer,
                target,
                final.get("action_char_span_half_open"),
            )
            thinking_span = _char_span_to_token_span(
                tokenizer,
                target,
                final.get("reasoning_char_span_half_open"),
            )
            metadata = {
                "example_id": row["task"]["task_id"],
                "schema_version": int(row["schema_version"]),
                "benchmark": row["benchmark"],
                "language": row["task"]["language"],
                "exercise": row["task"]["exercise"],
                "draft_passed": bool(row["draft_passed"]),
                "final_passed": bool(row["final_passed"]),
                "repair_attempted": bool(row["repair_attempted"]),
                "repair_improved": bool(row["repair_improved"]),
                "valid_final_boundary": bool(row["valid_final_boundary"]),
                "valid_final_action": bool(row["valid_final_action"]),
                "render_format": "frozen_generation_prompt",
                "action_char_span_half_open": final.get(
                    "action_char_span_half_open"
                ),
                "reasoning_char_span_half_open": final.get(
                    "reasoning_char_span_half_open"
                ),
            }
            examples.append(
                AiderExample(
                    prompt=prompt,
                    target=target,
                    metadata=metadata,
                    messages=tuple(history),
                    segments=segments,
                    sink_span=sink_span,
                    thinking_span=thinking_span,
                )
            )
            continue

        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("example_id", row.get("id", row_index))
        if row.get("length") is not None:
            metadata.setdefault("length", row.get("length"))

        if row.get("messages") is not None:
            messages = normalize_messages(row["messages"])
            prompt, target, segments = render_trajectory(messages, tokenizer=tokenizer)
            metadata.setdefault("schema_version", int(row.get("schema_version") or 2))
            metadata["render_format"] = "tokenizer_chat_template" if tokenizer is not None else "plain_fallback"
            metadata["assistant_turns"] = sum(message["role"] == "assistant" for message in messages)
            examples.append(
                AiderExample(
                    prompt=prompt,
                    target=target,
                    metadata=metadata,
                    messages=tuple(messages),
                    segments=segments,
                )
            )
            continue

        prompt = str(row.get("input") or row.get("prompt") or "")
        target = str(row.get("output") or row.get("target") or "")
        if not prompt.strip() or not target.strip():
            raise ValueError(
                f"Legacy Aider row {row_index} requires non-empty input/output (or prompt/target)."
            )
        metadata.setdefault("schema_version", 1)
        metadata.setdefault("render_format", "legacy_raw_prompt")
        examples.append(AiderExample(prompt=prompt, target=target, metadata=metadata))
    return examples
