"""Multimodal evaluation utilities for FlashTrace."""

from .datasets import MultimodalExample, load_examples, vqa_accuracy
from .metrics import (
    binary_iou,
    curve_auc,
    energy_in_mask,
    evidence_recall_at_fraction,
    pointing_game,
    xywh_boxes_to_mask,
)

__all__ = [
    "MultimodalExample",
    "binary_iou",
    "curve_auc",
    "energy_in_mask",
    "evidence_recall_at_fraction",
    "load_examples",
    "pointing_game",
    "vqa_accuracy",
    "xywh_boxes_to_mask",
]
