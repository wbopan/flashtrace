"""Model-independent metrics for visual attribution.

The functions in this module intentionally operate on NumPy arrays and saved
curves.  Attribution methods can therefore share one evaluator even when their
model hooks and image-token layouts differ.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _as_2d(values: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def curve_auc(scores: Sequence[float], fractions: Sequence[float] | None = None) -> float:
    """Return area under a perturbation curve on a normalized x-axis.

    ``scores`` should contain target-sequence scores measured while image
    evidence is progressively inserted or deleted.  If ``fractions`` is not
    supplied, points are assumed to be uniformly spaced from 0 to 1.  When it
    is supplied, the area is divided by the covered x-range, so the result is
    comparable across different numbers of perturbation steps.
    """

    y = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("scores must be a 1-D sequence with at least two points")
    if not np.all(np.isfinite(y)):
        raise ValueError("scores must contain only finite values")

    if fractions is None:
        x = np.linspace(0.0, 1.0, num=y.size, dtype=np.float64)
    else:
        x = np.asarray(fractions, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("fractions and scores must have the same length")
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError("fractions must be finite and strictly increasing")

    width = float(x[-1] - x[0])
    if width <= 0:
        raise ValueError("fractions must cover a non-zero range")
    return float(np.trapezoid(y, x=x) / width)


def pointing_game(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Return 1 when the maximum-attribution location is in the evidence mask."""

    attr = _as_2d(attribution, name="attribution").astype(np.float64, copy=False)
    mask = _as_2d(evidence_mask, name="evidence_mask").astype(bool, copy=False)
    if attr.shape != mask.shape:
        raise ValueError("attribution and evidence_mask must have the same shape")
    if not np.all(np.isfinite(attr)):
        raise ValueError("attribution must contain only finite values")
    if not np.any(mask):
        raise ValueError("evidence_mask must contain at least one positive location")

    return float(mask.flat[int(np.argmax(attr))])


def energy_in_mask(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Return the fraction of non-negative attribution mass inside evidence."""

    attr = _as_2d(attribution, name="attribution").astype(np.float64, copy=False)
    mask = _as_2d(evidence_mask, name="evidence_mask").astype(bool, copy=False)
    if attr.shape != mask.shape:
        raise ValueError("attribution and evidence_mask must have the same shape")
    if not np.all(np.isfinite(attr)):
        raise ValueError("attribution must contain only finite values")
    if not np.any(mask):
        raise ValueError("evidence_mask must contain at least one positive location")

    positive = np.clip(attr, a_min=0.0, a_max=None)
    total = float(positive.sum())
    if total == 0.0:
        return 0.0
    return float(positive[mask].sum() / total)


def evidence_recall_at_fraction(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
    fraction: float = 0.1,
) -> float:
    """Recall of evidence pixels among the top attribution fraction.

    This is the spatial analogue of FlashTrace's token Recovery Rate.  The top
    ``ceil(fraction * num_locations)`` locations are selected, then the metric
    reports what fraction of the ground-truth evidence mask they recover.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    attr = _as_2d(attribution, name="attribution").astype(np.float64, copy=False)
    mask = _as_2d(evidence_mask, name="evidence_mask").astype(bool, copy=False)
    if attr.shape != mask.shape:
        raise ValueError("attribution and evidence_mask must have the same shape")
    if not np.all(np.isfinite(attr)):
        raise ValueError("attribution must contain only finite values")

    evidence_count = int(mask.sum())
    if evidence_count == 0:
        raise ValueError("evidence_mask must contain at least one positive location")

    flat = attr.reshape(-1)
    selected_count = max(1, int(np.ceil(flat.size * fraction)))
    selected = np.argpartition(flat, flat.size - selected_count)[-selected_count:]
    recovered = int(mask.reshape(-1)[selected].sum())
    return float(recovered / evidence_count)


def binary_iou(
    prediction_mask: np.ndarray | Sequence[Sequence[bool]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Return intersection-over-union for two binary masks."""

    prediction = _as_2d(prediction_mask, name="prediction_mask").astype(bool, copy=False)
    evidence = _as_2d(evidence_mask, name="evidence_mask").astype(bool, copy=False)
    if prediction.shape != evidence.shape:
        raise ValueError("prediction_mask and evidence_mask must have the same shape")

    union = np.logical_or(prediction, evidence)
    if not np.any(union):
        return 1.0
    intersection = np.logical_and(prediction, evidence)
    return float(intersection.sum() / union.sum())


def xywh_boxes_to_mask(
    boxes: Iterable[Sequence[float]],
    height: int,
    width: int,
    *,
    normalized: bool = False,
) -> np.ndarray:
    """Rasterize ``(x, y, width, height)`` boxes into a boolean union mask.

    Boxes are clipped to image bounds.  Floating-point edges use floor for the
    top/left and ceil for the bottom/right, ensuring every touched pixel is
    included.  Normalized coordinates are interpreted relative to image size.
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        if len(box) != 4:
            raise ValueError("each box must contain exactly four values: x, y, w, h")
        x, y, box_width, box_height = (float(value) for value in box)
        if not np.all(np.isfinite([x, y, box_width, box_height])):
            raise ValueError("box coordinates must be finite")
        if box_width < 0 or box_height < 0:
            raise ValueError("box width and height must be non-negative")
        if normalized:
            x, box_width = x * width, box_width * width
            y, box_height = y * height, box_height * height

        left = max(0, min(width, int(np.floor(x))))
        top = max(0, min(height, int(np.floor(y))))
        right = max(0, min(width, int(np.ceil(x + box_width))))
        bottom = max(0, min(height, int(np.ceil(y + box_height))))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask
