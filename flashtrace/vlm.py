"""Small structural helpers shared by FlashTrace's VLM integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def is_vision_language_model(model: Any) -> bool:
    """Return whether ``model`` wraps a language decoder with a vision tower."""

    config = getattr(model, "config", None)
    model_body = getattr(model, "model", None)
    return bool(
        (config is not None and hasattr(config, "text_config") and hasattr(config, "vision_config"))
        or (
            model_body is not None
            and hasattr(model_body, "language_model")
            and hasattr(model_body, "visual")
        )
    )


def normalize_images(images: Any) -> list[Any]:
    """Normalize a single image or image sequence without treating paths as sequences."""

    if images is None:
        return []
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        return list(images)
    return [images]


def multimodal_messages(prompt: str, images: Any) -> list[dict[str, Any]]:
    """Build the Qwen-compatible one-turn message structure for image inputs."""

    content = [{"type": "image", "image": image} for image in normalize_images(images)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def contiguous_spans(indices: list[int]) -> list[tuple[int, int]]:
    """Compress sorted integer indices into inclusive spans."""

    if not indices:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            spans.append((start, previous))
            start = value
        previous = value
    spans.append((start, previous))
    return spans
