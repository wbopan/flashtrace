from __future__ import annotations

import json


def test_pages_build_is_static_and_contains_committed_samples(tmp_path):
    from demo.pages.build import build

    output = tmp_path / "site"
    samples = build(output)
    html = (output / "index.html").read_text(encoding="utf-8")
    manifest = json.loads(
        (output / "data" / "gallery.json").read_text(encoding="utf-8")
    )

    assert "FLASHTRACE_STATIC_DEMO = true" in html
    assert "/static/app.js" in html
    assert samples
    assert manifest["samples"]
    assert all(path.name.startswith("preset-") for path in samples)
    assert not (output / "api").exists()
