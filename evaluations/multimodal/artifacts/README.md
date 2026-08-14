# Strict attribution record artifacts

The row-level `attribution_records.jsonl` files are generated evaluation data,
not package source. They are distributed in the GitHub Release
[`multimodal-strict-attribution-v1`](https://github.com/wbopan/flashtrace/releases/tag/multimodal-strict-attribution-v1)
instead of Git LFS so ordinary clones do not transfer several gigabytes.

Restore every file to its original repository-relative path:

```bash
python evaluations/multimodal/download_strict_artifacts.py
```

Or restore one bundle:

```bash
python evaluations/multimodal/download_strict_artifacts.py --group formal
python evaluations/multimodal/download_strict_artifacts.py --group preview-final
python evaluations/multimodal/download_strict_artifacts.py --group pilot-smoke
```

The downloader uses only the Python standard library. It verifies the archive
size and SHA-256 before extraction, rejects unexpected or unsafe tar members,
and verifies every restored file against
[`strict-attribution-records-v1.json`](strict-attribution-records-v1.json).
Existing correct files are kept. A mismatched existing file is never replaced
unless `--force` is supplied.

The remaining compact summaries, tables, plots, manifests, and audit reports
stay in Git, so the published conclusions remain inspectable without fetching
the row-level artifacts.
