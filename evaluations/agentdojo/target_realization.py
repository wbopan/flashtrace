"""Stage planning and token provenance for target-realization experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence


FULL_BUDGET = "full"
TeacherForcingBudget = int | Literal["full"]


@dataclass(frozen=True)
class RealizationStage:
    """One point on a natural, prefix, or sparse-correction axis."""

    name: str
    family: Literal["natural", "prefix_forcing", "sparse_correction"]
    cumulative_natural_attempts: int
    teacher_forcing_tokens: TeacherForcingBudget | None = None
    sparse_correction_budget: TeacherForcingBudget | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_natural_budgets(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse strictly increasing cumulative natural-rollout budgets."""

    if isinstance(value, str):
        try:
            budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("Natural rollout budgets must be comma-separated integers.") from exc
    else:
        budgets = tuple(int(item) for item in value)
    if not budgets or any(item < 1 for item in budgets):
        raise ValueError("Natural rollout budgets must contain positive integers.")
    if budgets[0] != 1:
        raise ValueError("Natural rollout budgets must start at one greedy attempt.")
    if any(right <= left for left, right in zip(budgets, budgets[1:])):
        raise ValueError("Natural rollout budgets must be strictly increasing.")
    return budgets


def parse_teacher_forcing_budgets(
    value: str | Sequence[TeacherForcingBudget],
) -> tuple[TeacherForcingBudget, ...]:
    """Parse increasing token-prefix budgets with an optional final full stage."""

    raw_items = value.split(",") if isinstance(value, str) else value
    budgets: list[TeacherForcingBudget] = []
    for raw in raw_items:
        item = str(raw).strip().lower()
        if not item:
            continue
        if item == FULL_BUDGET:
            budgets.append(FULL_BUDGET)
            continue
        try:
            budget = int(item)
        except ValueError as exc:
            raise ValueError(
                "Teacher-forcing budgets must be positive integers or 'full'."
            ) from exc
        if budget < 1:
            raise ValueError("Teacher-forcing token budgets must be positive.")
        budgets.append(budget)
    numeric = [item for item in budgets if isinstance(item, int)]
    if any(right <= left for left, right in zip(numeric, numeric[1:])):
        raise ValueError("Teacher-forcing token budgets must be strictly increasing.")
    if budgets.count(FULL_BUDGET) > 1 or (
        FULL_BUDGET in budgets and budgets[-1] != FULL_BUDGET
    ):
        raise ValueError("The 'full' teacher-forcing stage must appear at most once, last.")
    return tuple(budgets)


def build_realization_stages(
    natural_budgets: Sequence[int],
    teacher_forcing_budgets: Sequence[TeacherForcingBudget] = (),
    sparse_correction_budgets: Sequence[TeacherForcingBudget] = (),
) -> tuple[RealizationStage, ...]:
    """Build one natural axis plus one explicitly selected intervention axis."""

    natural = parse_natural_budgets(natural_budgets)
    forced = parse_teacher_forcing_budgets(teacher_forcing_budgets)
    sparse = parse_teacher_forcing_budgets(sparse_correction_budgets)
    if forced and sparse:
        raise ValueError(
            "Prefix forcing and sparse correction must run as separate protocols."
        )
    stages = [
        RealizationStage(
            name=("natural_greedy" if budget == 1 else f"natural_sample_{budget}"),
            family="natural",
            cumulative_natural_attempts=budget,
        )
        for budget in natural
    ]
    max_natural = natural[-1]
    stages.extend(
        RealizationStage(
            name=f"prefix_forcing_{budget}",
            family="prefix_forcing",
            cumulative_natural_attempts=max_natural,
            teacher_forcing_tokens=budget,
        )
        for budget in forced
    )
    stages.extend(
        RealizationStage(
            name=f"sparse_correction_{budget}",
            family="sparse_correction",
            cumulative_natural_attempts=max_natural,
            sparse_correction_budget=budget,
        )
        for budget in sparse
    )
    return tuple(stages)


def natural_reasoning_prefix(completion: str) -> str | None:
    """Return a complete model-produced reasoning prefix and following whitespace."""

    boundary = completion.find("</think>")
    if boundary < 0:
        return None
    end = boundary + len("</think>")
    while end < len(completion) and completion[end].isspace():
        end += 1
    return completion[:end]


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(token_id)
        for token_id in tokenizer(text, add_special_tokens=False).input_ids
    ]


def decode_token_ids(tokenizer: Any, ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def teacher_forced_prefix(
    tokenizer: Any,
    action_text: str,
    budget: TeacherForcingBudget,
) -> tuple[str, list[int], int]:
    """Return the exact reference-action prefix selected by a token budget."""

    action_ids = token_ids(tokenizer, action_text)
    forced_count = len(action_ids) if budget == FULL_BUDGET else min(budget, len(action_ids))
    forced_ids = action_ids[:forced_count]
    return decode_token_ids(tokenizer, forced_ids), forced_ids, len(action_ids)


def completion_token_provenance(
    tokenizer: Any,
    completion: str,
    forced_char_span: tuple[int, int],
) -> list[str]:
    """Label each retokenized completion token as model- or reference-produced."""

    encoded = tokenizer(
        completion,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    start, end = forced_char_span
    return [
        "reference" if token_start < end and token_end > start else "model"
        for token_start, token_end in encoded["offset_mapping"]
    ]
