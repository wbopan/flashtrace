"""White-box visual attribution baselines for frozen Qwen3-VL responses.

All methods in this module attribute an already frozen response. They do not
generate a new answer, so their maps are directly comparable with the visual
leave-one-region-out reference used by the multimodal evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flashtrace.lrp_patches import lrp_context


@dataclass
class FrozenVLMInputs:
    """Teacher-forced multimodal inputs plus attribution index metadata."""

    inputs: dict[str, torch.Tensor]
    prompt_length: int
    target_token_ids: torch.Tensor
    target_offsets: torch.Tensor
    predictor_positions: torch.Tensor
    visual_indices: torch.Tensor
    visual_grid_shape: tuple[int, int]
    raw_image_grid_thw: tuple[int, int, int]
    spatial_merge_size: int


@dataclass
class VisualBaselineOutput:
    """Native visual-token attribution grid and method metadata."""

    grid: list[list[float]]
    metadata: dict[str, Any]


def _as_model_inputs(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    if hasattr(batch, "to"):
        batch = batch.to(device)
    return {
        key: value
        for key, value in dict(batch).items()
        if isinstance(value, torch.Tensor)
    }


def prepare_frozen_vlm_inputs(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    target: str,
    *,
    output_span: tuple[int, int] | None = None,
) -> FrozenVLMInputs:
    """Build one full prompt-plus-target Qwen3-VL teacher-forcing batch."""

    device = model.device
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _as_model_inputs(batch, device)
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Visual baselines currently require batch size 1")
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise ValueError("Visual baselines require one image and image_grid_thw")
    if int(inputs["image_grid_thw"].shape[0]) != 1:
        raise ValueError("Visual baselines currently require exactly one image")

    prompt_length = int(inputs["input_ids"].shape[1])
    target_ids = processor.tokenizer(
        target,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    if target_ids.shape[0] != 1 or target_ids.shape[1] == 0:
        raise ValueError("The frozen target must contain at least one token")

    inputs["input_ids"] = torch.cat((inputs["input_ids"], target_ids), dim=1)
    target_length = int(target_ids.shape[1])
    inputs["attention_mask"] = torch.cat(
        (
            inputs.get(
                "attention_mask",
                torch.ones(
                    (1, prompt_length),
                    dtype=torch.long,
                    device=device,
                ),
            ),
            torch.ones((1, target_length), dtype=torch.long, device=device),
        ),
        dim=1,
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            (
                inputs["mm_token_type_ids"],
                torch.zeros(
                    (1, target_length),
                    dtype=inputs["mm_token_type_ids"].dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
    # These fields, when emitted by a processor, only describe the prompt.
    inputs.pop("position_ids", None)
    inputs.pop("cache_position", None)

    if output_span is None:
        start, end = 0, target_length - 1
    else:
        start, end = (int(output_span[0]), int(output_span[1]))
    if start < 0 or end < start or end >= target_length:
        raise ValueError(
            f"Invalid output_span {output_span!r} for {target_length} target tokens"
        )
    offsets = torch.arange(start, end + 1, dtype=torch.long, device=device)
    predictors = prompt_length + offsets - 1

    image_token_id = int(model.config.image_token_id)
    visual_indices = torch.nonzero(
        inputs["input_ids"][0, :prompt_length] == image_token_id,
        as_tuple=False,
    ).flatten()
    if visual_indices.numel() == 0:
        raise ValueError("No image placeholder tokens found in the prompt")

    grid_values = inputs["image_grid_thw"][0].detach().cpu().tolist()
    frames, raw_height, raw_width = (int(value) for value in grid_values)
    merge = int(model.config.vision_config.spatial_merge_size)
    if frames != 1:
        raise ValueError(f"Expected a still image, got {frames} temporal frames")
    if raw_height % merge or raw_width % merge:
        raise ValueError(
            f"Image grid {(raw_height, raw_width)} is not divisible by merge={merge}"
        )
    visual_shape = (raw_height // merge, raw_width // merge)
    if visual_indices.numel() != visual_shape[0] * visual_shape[1]:
        raise ValueError(
            f"Found {visual_indices.numel()} visual tokens for grid {visual_shape}"
        )

    return FrozenVLMInputs(
        inputs=inputs,
        prompt_length=prompt_length,
        target_token_ids=target_ids[0],
        target_offsets=offsets,
        predictor_positions=predictors,
        visual_indices=visual_indices,
        visual_grid_shape=visual_shape,
        raw_image_grid_thw=(frames, raw_height, raw_width),
        spatial_merge_size=merge,
    )


def _forward_kwargs(prepared: FrozenVLMInputs) -> dict[str, torch.Tensor]:
    return dict(prepared.inputs)


def _selected_logits(
    logits: torch.Tensor, prepared: FrozenVLMInputs
) -> torch.Tensor:
    token_ids = prepared.target_token_ids[prepared.target_offsets]
    return logits[0, prepared.predictor_positions, token_ids]


def _selected_mean_logprob(
    logits: torch.Tensor, prepared: FrozenVLMInputs
) -> torch.Tensor:
    token_ids = prepared.target_token_ids[prepared.target_offsets]
    rows = logits[0, prepared.predictor_positions].float()
    return rows.log_softmax(dim=-1).gather(1, token_ids[:, None]).mean()


def _grid_from_visual_scores(
    scores: torch.Tensor, prepared: FrozenVLMInputs
) -> list[list[float]]:
    expected = prepared.visual_grid_shape[0] * prepared.visual_grid_shape[1]
    flat = scores.detach().float().cpu().reshape(-1)
    if flat.numel() != expected:
        raise ValueError(f"Expected {expected} visual scores, got {flat.numel()}")
    return flat.reshape(prepared.visual_grid_shape).tolist()


def _rollout_rows(
    attentions: tuple[torch.Tensor, ...],
    prepared: FrozenVLMInputs,
    *,
    gradients: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    sequence_length = int(prepared.inputs["input_ids"].shape[1])
    rows = torch.eye(
        sequence_length,
        device=attentions[0].device,
        dtype=torch.float32,
    )[prepared.predictor_positions.to(attentions[0].device)]
    identity = torch.eye(
        sequence_length,
        device=attentions[0].device,
        dtype=torch.float32,
    )

    if gradients is None:
        layer_maps = [
            attention.detach().float().mean(dim=1)[0] for attention in attentions
        ]
    else:
        layer_maps = [
            (attention.float() * gradient.float())
            .clamp_min(0)
            .mean(dim=1)[0]
            for attention, gradient in zip(attentions, gradients)
        ]

    return _rollout_from_layer_maps(layer_maps, rows=rows, identity=identity)


def _rollout_from_layer_maps(
    layer_maps: list[torch.Tensor],
    *,
    rows: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    for layer_map in reversed(layer_maps):
        layer_map = layer_map.to(rows.device)
        transition = layer_map + identity
        transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(transition.dtype).eps
        )
        rows = rows @ transition
    return rows


def attention_rollout(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    target: str,
    *,
    output_span: tuple[int, int] | None = None,
) -> VisualBaselineOutput:
    """Mean-head decoder attention rollout to fused visual tokens."""

    prepared = prepare_frozen_vlm_inputs(
        model, processor, messages, target, output_span=output_span
    )
    with torch.no_grad():
        outputs = model(
            **_forward_kwargs(prepared),
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    attentions = tuple(outputs.attentions or ())
    if not attentions:
        raise RuntimeError(
            "The model returned no attentions; load it with attn_implementation='eager'"
        )
    rows = _rollout_rows(attentions, prepared)
    visual_scores = rows[:, prepared.visual_indices.to(rows.device)].mean(dim=0)
    return VisualBaselineOutput(
        grid=_grid_from_visual_scores(visual_scores, prepared),
        metadata={
            "formula": "mean-head attention rollout with identity residuals",
            "objective": "frozen target span",
            "layers": len(attentions),
            "attributed_tokens": int(prepared.target_offsets.numel()),
            "visual_grid_shape": list(prepared.visual_grid_shape),
        },
    )


def grad_attention(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    target: str,
    *,
    output_span: tuple[int, int] | None = None,
) -> VisualBaselineOutput:
    """Positive Grad×Attention rollout for frozen target-token logits."""

    prepared = prepare_frozen_vlm_inputs(
        model, processor, messages, target, output_span=output_span
    )
    model.zero_grad(set_to_none=True)
    full_ids = prepared.inputs["input_ids"]
    attention_mask = prepared.inputs["attention_mask"]
    image_grid = prepared.inputs["image_grid_thw"]
    mm_token_types = prepared.inputs.get("mm_token_type_ids")
    # Decoder Grad×Attention needs fused visual-token attentions, not the
    # quadratic internal attention maps of the 2MP vision encoder. Compute the
    # image features once without gradients, then attribute the text decoder.
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(full_ids)
        image_outputs = model.model.get_image_features(
            prepared.inputs["pixel_values"],
            image_grid,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            text_embeds.device, text_embeds.dtype
        )
        image_mask, _ = model.model.get_placeholder_mask(
            full_ids,
            inputs_embeds=text_embeds,
            image_features=image_embeds,
        )
        fused_embeds = text_embeds.masked_scatter(image_mask, image_embeds)
        visual_mask = image_mask[..., 0]
        position_ids = model.model.compute_3d_position_ids(
            input_ids=full_ids,
            image_grid_thw=image_grid,
            video_grid_thw=None,
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_types,
        )
        deepstack_features = [
            feature.detach() for feature in image_outputs.deepstack_features
        ]
    del image_outputs, image_embeds, text_embeds

    checkpointing_was_enabled = bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    training_was_enabled = bool(model.training)
    if not checkpointing_was_enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # Checkpointing is active only in training mode. Qwen3-VL has no active
    # dropout in this configuration, so the frozen response remains unchanged.
    model.train()
    hooks = []
    try:
        decoder_outputs = model.model.language_model(
            input_ids=None,
            inputs_embeds=fused_embeds.detach(),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack_features,
            return_dict=True,
        )
        attentions = tuple(decoder_outputs.attentions or ())
        if not attentions:
            raise RuntimeError(
                "The model returned no attentions; load it with "
                "attn_implementation='eager'"
            )
        layer_maps: list[torch.Tensor | None] = [None] * len(attentions)
        for index, attention in enumerate(attentions):
            def capture(
                gradient: torch.Tensor, *, layer_index: int = index
            ) -> torch.Tensor:
                # A full retained gradient for every head/layer exceeds 72 GB
                # on 2MP document inputs. Reduce Grad×Attention over heads in
                # the hook, then let the transient full gradient be freed.
                layer_maps[layer_index] = (
                    (attentions[layer_index].detach().float() * gradient.float())
                    .clamp_min(0)
                    .mean(dim=1)[0]
                    .cpu()
                )
                return gradient

            hooks.append(attention.register_hook(capture))
        # Project only the predictor rows. Materializing vocab logits for every
        # document token costs more than 1 GB on long Wiki-VISA responses and
        # does not contribute to the output-only objective.
        predictor_positions = prepared.predictor_positions.to(
            decoder_outputs.last_hidden_state.device
        )
        selected_hidden = decoder_outputs.last_hidden_state[
            0, predictor_positions
        ]
        selected_logits = model.lm_head(selected_hidden)
        selected_ids = prepared.target_token_ids[
            prepared.target_offsets
        ].to(selected_logits.device)
        objective = selected_logits[
            torch.arange(selected_logits.shape[0], device=selected_logits.device),
            selected_ids,
        ].sum()
        objective.backward()
    finally:
        for hook in hooks:
            hook.remove()
        if not checkpointing_was_enabled:
            model.gradient_checkpointing_disable()
        if not training_was_enabled:
            model.eval()
    if any(layer_map is None for layer_map in layer_maps):
        raise RuntimeError("Could not capture gradients for all decoder attentions")
    sequence_length = int(prepared.inputs["input_ids"].shape[1])
    rows = torch.eye(
        sequence_length,
        device=attentions[0].device,
        dtype=torch.float32,
    )[prepared.predictor_positions.to(attentions[0].device)]
    identity = torch.eye(
        sequence_length,
        device=attentions[0].device,
        dtype=torch.float32,
    )
    rows = _rollout_from_layer_maps(
        [layer_map for layer_map in layer_maps if layer_map is not None],
        rows=rows,
        identity=identity,
    )
    visual_scores = rows[:, prepared.visual_indices.to(rows.device)].mean(dim=0)
    model.zero_grad(set_to_none=True)
    return VisualBaselineOutput(
        grid=_grid_from_visual_scores(visual_scores, prepared),
        metadata={
            "formula": "positive (gradient * attention), mean heads, residual rollout",
            "scope": "Qwen3-VL text decoder over fused visual tokens",
            "objective": "sum of frozen target-token logits",
            "objective_value": float(objective.detach().float().cpu()),
            "layers": len(attentions),
            "attributed_tokens": int(prepared.target_offsets.numel()),
            "visual_grid_shape": list(prepared.visual_grid_shape),
        },
    )


def _merge_raw_patch_scores(
    raw_scores: torch.Tensor, prepared: FrozenVLMInputs
) -> torch.Tensor:
    frames, raw_height, raw_width = prepared.raw_image_grid_thw
    merge = prepared.spatial_merge_size
    expected = frames * raw_height * raw_width
    if raw_scores.numel() != expected:
        raise ValueError(
            f"pixel_values produced {raw_scores.numel()} patch scores; "
            f"image_grid_thw expects {expected}"
        )
    # Qwen-VL processors order raw patches by merged block, with the merge^2
    # patches for one fused visual token contiguous.
    merged = raw_scores.reshape(
        frames,
        raw_height // merge,
        raw_width // merge,
        merge * merge,
    ).sum(dim=-1)
    return merged.sum(dim=0)


def visual_integrated_gradients(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    target: str,
    *,
    output_span: tuple[int, int] | None = None,
    steps: int = 20,
) -> VisualBaselineOutput:
    """Integrated gradients from zero to the processed image-patch tensor."""

    if steps < 2:
        raise ValueError("Visual IG requires at least two integration steps")
    prepared = prepare_frozen_vlm_inputs(
        model, processor, messages, target, output_span=output_span
    )
    actual = prepared.inputs["pixel_values"].detach()
    baseline = torch.zeros_like(actual)
    delta = actual - baseline
    accumulated = torch.zeros_like(actual, dtype=torch.float32)
    endpoint_scores: list[float] = []

    checkpointing_was_enabled = bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    training_was_enabled = bool(model.training)
    if not checkpointing_was_enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # Transformers activates decoder checkpointing in training mode. Qwen3-VL
    # has no active dropout in this configuration, and the response is frozen.
    model.train()
    try:
        for step_index, alpha in enumerate(
            torch.linspace(0.0, 1.0, steps, device=actual.device)
        ):
            pixels = (baseline + alpha * delta).detach().requires_grad_(True)
            inputs = _forward_kwargs(prepared)
            inputs["pixel_values"] = pixels
            outputs = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
            objective = _selected_mean_logprob(outputs.logits, prepared)
            gradient = torch.autograd.grad(objective, pixels)[0]
            weight = 0.5 if step_index in {0, steps - 1} else 1.0
            accumulated += weight * gradient.detach().float()
            if step_index in {0, steps - 1}:
                endpoint_scores.append(float(objective.detach().cpu()))
    finally:
        if not checkpointing_was_enabled:
            model.gradient_checkpointing_disable()
        if not training_was_enabled:
            model.eval()

    mean_gradient = accumulated / float(steps - 1)
    attribution = delta.float() * mean_gradient
    raw_patch_scores = attribution.reshape(attribution.shape[0], -1).sum(dim=-1)
    visual_scores = _merge_raw_patch_scores(raw_patch_scores, prepared)
    score_delta = endpoint_scores[1] - endpoint_scores[0]
    attribution_sum = float(attribution.sum().detach().cpu())
    return VisualBaselineOutput(
        grid=_grid_from_visual_scores(visual_scores, prepared),
        metadata={
            "formula": "integrated gradients on processed pixel patches",
            "baseline": "zero processed-pixel tensor",
            "objective": "mean frozen target-token log probability",
            "steps": steps,
            "attributed_tokens": int(prepared.target_offsets.numel()),
            "score_delta": score_delta,
            "attribution_sum": attribution_sum,
            "completeness_residual": attribution_sum - score_delta,
            "visual_grid_shape": list(prepared.visual_grid_shape),
        },
    )


def qwen3_vl_attnlrp(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    target: str,
    *,
    output_span: tuple[int, int] | None = None,
) -> VisualBaselineOutput:
    """AttnLRP over Qwen3-VL fused image tokens, including DeepStack paths."""

    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != "qwen3_vl":
        raise ValueError(
            f"Qwen3-VL AttnLRP requires model_type='qwen3_vl', got {model_type!r}"
        )
    prepared = prepare_frozen_vlm_inputs(
        model, processor, messages, target, output_span=output_span
    )
    full_ids = prepared.inputs["input_ids"]
    attention_mask = prepared.inputs["attention_mask"]
    image_grid = prepared.inputs["image_grid_thw"]
    mm_token_types = prepared.inputs.get("mm_token_type_ids")

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(full_ids)
        image_outputs = model.model.get_image_features(
            prepared.inputs["pixel_values"],
            image_grid,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            text_embeds.device, text_embeds.dtype
        )
        image_mask, _ = model.model.get_placeholder_mask(
            full_ids,
            inputs_embeds=text_embeds,
            image_features=image_embeds,
        )
        fused_embeds = text_embeds.masked_scatter(image_mask, image_embeds)
        visual_mask = image_mask[..., 0]
        position_ids = model.model.compute_3d_position_ids(
            input_ids=full_ids,
            image_grid_thw=image_grid,
            video_grid_thw=None,
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_types,
        )

    fused_leaf = fused_embeds.detach().requires_grad_(True)
    deepstack_leaves = [
        feature.detach().requires_grad_(True)
        for feature in image_outputs.deepstack_features
    ]
    model.zero_grad(set_to_none=True)
    checkpointing_was_enabled = bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    training_was_enabled = bool(model.training)
    if not checkpointing_was_enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # Decoder checkpointing is active only in training mode. The LRP context
    # patches dropout to identity, so this remains deterministic.
    model.train()
    try:
        with lrp_context(model, "qwen3_vl"):
            outputs = model.model.language_model(
                input_ids=None,
                inputs_embeds=fused_leaf,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                visual_pos_masks=visual_mask,
                deepstack_visual_embeds=deepstack_leaves,
                return_dict=True,
            )
            # Materializing [full_sequence, vocabulary] logits can add
            # hundreds of MiB on long VizWiz responses. The registered
            # objective uses only predictor rows, so project exactly those
            # hidden states through the unchanged LM head.
            predictor_positions = prepared.predictor_positions.to(
                outputs.last_hidden_state.device
            )
            selected_hidden = outputs.last_hidden_state[
                0, predictor_positions
            ]
            selected_logits = model.lm_head(selected_hidden)
            target_ids = prepared.target_token_ids[
                prepared.target_offsets
            ].to(selected_logits.device)
            objective = selected_logits.gather(
                1, target_ids[:, None]
            ).sum()
            gradients = torch.autograd.grad(
                objective,
                [fused_leaf, *deepstack_leaves],
                allow_unused=True,
            )
    finally:
        if not checkpointing_was_enabled:
            model.gradient_checkpointing_disable()
        if not training_was_enabled:
            model.eval()

    fused_gradient = gradients[0]
    if fused_gradient is None:
        raise RuntimeError("AttnLRP did not return relevance for fused embeddings")
    visual_scores = (fused_leaf * fused_gradient).sum(dim=-1)[visual_mask]
    for feature, gradient in zip(deepstack_leaves, gradients[1:]):
        if gradient is not None:
            visual_scores = visual_scores + (feature * gradient).sum(dim=-1)
    model.zero_grad(set_to_none=True)
    return VisualBaselineOutput(
        grid=_grid_from_visual_scores(visual_scores, prepared),
        metadata={
            "formula": "AttnLRP gradient rules on the Qwen3-VL text decoder",
            "scope": "fused visual tokens plus all DeepStack injection paths",
            "objective": "sum of frozen target-token logits",
            "objective_value": float(objective.detach().float().cpu()),
            "deepstack_paths": len(deepstack_leaves),
            "attributed_tokens": int(prepared.target_offsets.numel()),
            "visual_grid_shape": list(prepared.visual_grid_shape),
        },
    )
