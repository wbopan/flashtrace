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


def _patch_evidence_statistics(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return patch scores plus evidence and pixel area for every patch.

    Evidence stays at its native pixel resolution.  Each evidence pixel is
    assigned to exactly one visual patch, so no interpolation or partial-patch
    top-k selection is introduced by evaluation.
    """

    attr = _as_2d(attribution, name="attribution").astype(np.float64, copy=False)
    mask = _as_2d(evidence_mask, name="evidence_mask").astype(bool, copy=False)
    if not np.all(np.isfinite(attr)):
        raise ValueError("attribution must contain only finite values")
    if not np.any(mask):
        raise ValueError("evidence_mask must contain at least one positive location")

    grid_height, grid_width = attr.shape
    height, width = mask.shape
    rows = np.minimum(grid_height - 1, np.arange(height) * grid_height // height)
    columns = np.minimum(grid_width - 1, np.arange(width) * grid_width // width)
    patch_ids = rows[:, None] * grid_width + columns[None, :]
    patch_count = grid_height * grid_width
    evidence = np.bincount(
        patch_ids.reshape(-1),
        weights=mask.reshape(-1).astype(np.float64),
        minlength=patch_count,
    )
    areas = np.bincount(patch_ids.reshape(-1), minlength=patch_count).astype(
        np.float64
    )
    return attr.reshape(-1), evidence, areas


def _expected_top_weights(
    scores: np.ndarray, weights: np.ndarray, selected_count: int
) -> tuple[float, float]:
    """Expected selected weight and area under a uniform cutoff-tie break."""

    if not 1 <= selected_count <= scores.size:
        raise ValueError("selected_count must be within the patch count")
    threshold = float(np.partition(scores, scores.size - selected_count)[-selected_count])
    above = scores > threshold
    tied = scores == threshold
    remaining = selected_count - int(above.sum())
    tie_count = int(tied.sum())
    tie_fraction = remaining / tie_count
    selected_weight = float(weights[above].sum()) + tie_fraction * float(
        weights[tied].sum()
    )
    return selected_weight, tie_fraction


def patch_recovery_at_fraction(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
    fraction: float = 0.1,
) -> float:
    """Evidence recovery after selecting a top fraction of complete patches.

    If the cutoff score is tied, evidence in that score group receives its
    expected credit under a uniform tie break.  This preserves the exact patch
    budget without introducing flatten-order bias.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    scores, evidence, _ = _patch_evidence_statistics(attribution, evidence_mask)
    selected_count = max(1, int(np.ceil(scores.size * fraction)))
    recovered, _ = _expected_top_weights(scores, evidence, selected_count)
    return float(recovered / evidence.sum())


def patch_energy_in_mask(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Positive patch-attribution mass weighted by within-patch GT overlap."""

    scores, evidence, areas = _patch_evidence_statistics(attribution, evidence_mask)
    positive = np.clip(scores, 0.0, None)
    total = float(positive.sum())
    if total == 0.0:
        return 0.0
    overlap_fraction = np.divide(
        evidence,
        areas,
        out=np.zeros_like(evidence),
        where=areas > 0,
    )
    return float(np.dot(positive, overlap_fraction) / total)


def patch_pointing_game(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Expected max-patch hit rate with uniform handling of maximum ties."""

    scores, evidence, _ = _patch_evidence_statistics(attribution, evidence_mask)
    maxima = scores == scores.max()
    return float(np.mean(evidence[maxima] > 0.0))


def patch_evidence_rank_auc(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Pixel-weighted probability that an evidence patch outranks background."""

    scores, evidence, areas = _patch_evidence_statistics(attribution, evidence_mask)
    background = areas - evidence
    positive_total = float(evidence.sum())
    negative_total = float(background.sum())
    if positive_total == 0.0 or negative_total == 0.0:
        raise ValueError("evidence_mask must contain both evidence and background")

    order = np.argsort(scores, kind="mergesort")
    negative_below = 0.0
    favorable_pairs = 0.0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[order[end]] == scores[order[start]]:
            end += 1
        group = order[start:end]
        positive_group = float(evidence[group].sum())
        negative_group = float(background[group].sum())
        favorable_pairs += positive_group * (
            negative_below + 0.5 * negative_group
        )
        negative_below += negative_group
        start = end
    return favorable_pairs / (positive_total * negative_total)


def patch_top_evidence_iou(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Expected IoU at a whole-patch budget matching the GT area fraction."""

    scores, evidence, areas = _patch_evidence_statistics(attribution, evidence_mask)
    selected_count = max(
        1,
        int(np.ceil(scores.size * float(evidence.sum() / areas.sum()))),
    )
    intersection, tie_fraction = _expected_top_weights(
        scores, evidence, selected_count
    )
    threshold = float(np.partition(scores, scores.size - selected_count)[-selected_count])
    above = scores > threshold
    tied = scores == threshold
    selected_area = float(areas[above].sum()) + tie_fraction * float(
        areas[tied].sum()
    )
    union = selected_area + float(evidence.sum()) - intersection
    return float(intersection / union) if union > 0.0 else 1.0


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


def evidence_rank_auc(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """Probability that a random evidence pixel outranks a background pixel.

    This is the tie-aware ROC AUC computed directly from attribution ranks.  It
    complements fixed top-q recovery when evidence occupies a large fraction
    of the image.
    """

    attr = _as_2d(attribution, name="attribution").astype(np.float64, copy=False)
    mask = _as_2d(evidence_mask, name="evidence_mask").astype(bool, copy=False)
    if attr.shape != mask.shape:
        raise ValueError("attribution and evidence_mask must have the same shape")
    if not np.all(np.isfinite(attr)):
        raise ValueError("attribution must contain only finite values")
    positives = attr[mask]
    negatives = attr[~mask]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("evidence_mask must contain both evidence and background")

    values = np.concatenate((positives, negatives))
    labels = np.concatenate(
        (
            np.ones(positives.size, dtype=np.int8),
            np.zeros(negatives.size, dtype=np.int8),
        )
    )
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    count_positive = float(positives.size)
    count_negative = float(negatives.size)
    return (
        positive_rank_sum - count_positive * (count_positive + 1.0) / 2.0
    ) / (count_positive * count_negative)


def top_evidence_iou(
    attribution: np.ndarray | Sequence[Sequence[float]],
    evidence_mask: np.ndarray | Sequence[Sequence[bool]],
) -> float:
    """IoU after selecting exactly as many top pixels as the evidence area."""

    attr = _as_2d(attribution, name="attribution").astype(np.float64, copy=False)
    mask = _as_2d(evidence_mask, name="evidence_mask").astype(bool, copy=False)
    if attr.shape != mask.shape:
        raise ValueError("attribution and evidence_mask must have the same shape")
    if not np.all(np.isfinite(attr)):
        raise ValueError("attribution must contain only finite values")
    count = int(mask.sum())
    if count <= 0:
        raise ValueError("evidence_mask must contain at least one positive location")
    flat = attr.reshape(-1)
    selected = np.argpartition(flat, flat.size - count)[-count:]
    prediction = np.zeros(flat.size, dtype=bool)
    prediction[selected] = True
    return binary_iou(prediction.reshape(mask.shape), mask)


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


def xyxy_boxes_to_mask(
    boxes: Iterable[Sequence[float]],
    height: int,
    width: int,
    *,
    normalized: bool = False,
) -> np.ndarray:
    """Rasterize ``(x1, y1, x2, y2)`` boxes into a boolean union mask."""

    converted = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("each box must contain exactly four values: x1, y1, x2, y2")
        x1, y1, x2, y2 = (float(value) for value in box)
        if not np.all(np.isfinite([x1, y1, x2, y2])):
            raise ValueError("box coordinates must be finite")
        if x2 < x1 or y2 < y1:
            raise ValueError("box bottom/right must not precede top/left")
        converted.append((x1, y1, x2 - x1, y2 - y1))
    return xywh_boxes_to_mask(
        converted,
        height=height,
        width=width,
        normalized=normalized,
    )
