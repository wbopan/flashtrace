from pathlib import Path

from exp.exp4.polyglot import (
    LANGUAGES,
    discover_tasks,
    parse_whole_file_edit,
    test_feedback as build_test_feedback,
)
from exp.exp4.run_local import _completion_spans, _stratified_selection


ROOT = Path(__file__).parents[1]
POLYGLOT = ROOT / "data" / "external" / "aider-polyglot"


def test_discover_pinned_polyglot_inventory():
    tasks = discover_tasks(POLYGLOT)

    assert len(tasks) == 225
    assert {task.language for task in tasks} == set(LANGUAGES)
    proverb = next(task for task in tasks if task.task_id == "polyglot-python-proverb")
    assert proverb.solution_files == ("proverb.py",)
    assert "For want of a horseshoe nail" in proverb.prompt


def test_parse_whole_file_edit_tracks_strict_format_and_single_file_fallback():
    strict = parse_whole_file_edit(
        "src/main.py\n```python\nprint('ok')\n```",
        ["src/main.py"],
    )
    assert strict.files == {"src/main.py": "print('ok')\n"}
    assert strict.format_compliant
    assert strict.errors == ()

    fallback = parse_whole_file_edit("```python\nprint('ok')\n```", ["main.py"])
    assert fallback.files == {"main.py": "print('ok')\n"}
    assert not fallback.format_compliant
    assert "omitted" in fallback.errors[0]


def test_feedback_is_deterministically_bounded():
    class Result:
        output = "x" * 20_000

    feedback = build_test_feedback(Result(), ["main.py"], max_characters=1_000)
    assert "deterministically truncated" in feedback
    assert len(feedback) < 1_500


def test_stratified_selection_covers_all_languages():
    tasks = discover_tasks(POLYGLOT)
    selected = _stratified_selection(tasks, 30, seed=17)

    assert len(selected) == 30
    assert {task.language for task in selected} == set(LANGUAGES)
    assert all(
        sum(task.language == language for task in selected) == 5
        for language in LANGUAGES
    )
    assert selected == _stratified_selection(tasks, 30, seed=17)


def test_completion_spans_separate_reasoning_from_action():
    completion = "<think>\nreason\n</think>\n\nmain.py\n```\npass\n```"
    spans = _completion_spans(completion)

    reasoning = spans["reasoning_char_span_half_open"]
    action = spans["action_char_span_half_open"]
    assert completion[slice(*reasoning)].strip() == "reason"
    assert completion[slice(*action)].startswith("main.py")
    assert spans["reasoning_boundary"] == "explicit_open_and_close"


def test_completion_spans_support_implicit_open_and_reject_missing_close():
    implicit = "reason\n</think>\nmain.py\n```\npass\n```"
    spans = _completion_spans(implicit)
    assert implicit[slice(*spans["reasoning_char_span_half_open"])].strip() == "reason"
    assert implicit[slice(*spans["action_char_span_half_open"])].startswith("main.py")
    assert spans["reasoning_boundary"] == "implicit_open_explicit_close"

    truncated = _completion_spans("<think>\nstill reasoning")
    assert truncated["reasoning_boundary"] == "missing_close"
    assert truncated["action_char_span_half_open"] is None
