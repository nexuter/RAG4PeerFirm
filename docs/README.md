# RAG4PeerFirm Technical Docs

This folder contains implementation-level documentation for the three main modules:

- [Quickstart (First-Time Users)](./quickstart.md)
- [Downloader Module](./downloader.md)
- [Vector DB Builder Module](./vdbbuilder.md)
- [Peer Finder Module](./peerfinder.md)
- [Pipeline Walkthrough](./pipeline_walkthrough.md)

Recommended reading order for new users:

1. `quickstart.md` (first successful run)
2. `pipeline_walkthrough.md` (end-to-end flow)
3. `downloader.md` (how raw filings are collected)
4. `vdbbuilder.md` (how vectors/indices are built)
5. `peerfinder.md` (how peer sets are computed)

---

## Terminology

- `CIK`: SEC company identifier (10-digit zero-padded string).
- `Item`: 10-K section id (e.g., `1A`, `7`, `7A`).
- `Scope`:
  - `all`: full item text from `*_item.json`.
  - `heading`: heading-only text from `*_str.json`.
  - `body`: body-only text from `*_str.json`.
- `Pooled vector`: item embedding formed from unit embeddings.
- `Residual vector`: item-specific component after removing common direction.
- `Common similarity`: pooled cosine similarity.
- `Specific similarity`: residual cosine similarity.

## Testing Notes

- Integration tests live in `tests/test_integration_pipeline.py`.
- You can run only builder-related tests with:
  - `python -m pytest tests/test_integration_pipeline.py -k vdbbuilder`
- `pytest.ini` is ignored in this repo; if you want custom marker config locally, create your own `pytest.ini`.
