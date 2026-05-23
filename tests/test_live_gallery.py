from __future__ import annotations

import pytest


def _record(title="Sample"):
    return {
        "title": title,
        "model": "tiny-qwen2",
        "method": "flashtrace",
        "hops": 1,
        "prompt": "Question: capital of France?",
        "generated_text": "Paris",
        "target_span": "0:0",
        "reasoning_span": "",
        "render_model": {"phase": "traced", "views": [], "active_view": 0},
        "trace_json": {"method": "flashtrace"},
    }


def test_save_then_list_then_load_roundtrip(tmp_path):
    from demo.live import gallery

    summary = gallery.save_sample(tmp_path, _record("First"))
    assert summary["title"] == "First"
    assert summary["id"]
    assert summary["created_at"]
    assert "render_model" not in summary
    assert summary["prompt_preview"].startswith("Question:")

    listed = gallery.list_samples(tmp_path)
    assert [item["id"] for item in listed] == [summary["id"]]

    full = gallery.load_sample(tmp_path, summary["id"])
    assert full["render_model"] == {"phase": "traced", "views": [], "active_view": 0}
    assert full["trace_json"] == {"method": "flashtrace"}


def test_list_sorted_newest_first(tmp_path):
    from demo.live import gallery

    a = gallery.save_sample(tmp_path, _record("A"))
    b = gallery.save_sample(tmp_path, _record("B"))
    ids = [item["id"] for item in gallery.list_samples(tmp_path)]
    assert set(ids) == {a["id"], b["id"]}
    # created_at descending; ids are unique per save.
    created = [item["created_at"] for item in gallery.list_samples(tmp_path)]
    assert created == sorted(created, reverse=True)


def test_delete_removes_sample(tmp_path):
    from demo.live import gallery

    summary = gallery.save_sample(tmp_path, _record())
    gallery.delete_sample(tmp_path, summary["id"])
    assert gallery.list_samples(tmp_path) == []
    with pytest.raises(KeyError):
        gallery.load_sample(tmp_path, summary["id"])


def test_list_on_missing_dir_returns_empty(tmp_path):
    from demo.live import gallery

    assert gallery.list_samples(tmp_path / "nope") == []


def test_save_missing_field_raises(tmp_path):
    from demo.live import gallery

    record = _record()
    del record["render_model"]
    with pytest.raises(ValueError):
        gallery.save_sample(tmp_path, record)


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", "x.json"])
def test_invalid_id_rejected(tmp_path, bad_id):
    from demo.live import gallery

    with pytest.raises(ValueError):
        gallery.load_sample(tmp_path, bad_id)
