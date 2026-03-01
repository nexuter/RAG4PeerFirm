# Pipeline Walkthrough

This document explains the full workflow from SEC download to peer-firm output.

## 1) Download filings

Use `script/downloader.py` to fetch filings and metadata from EDGAR.

Core output layout:

```text
sec_filings/
  <CIK>/
    <YEAR>/
      <FORM>/                     # e.g., 10-K
        <CIK>_<YEAR>_<FORM>.htm
        <CIK>_<YEAR>_<FORM>_meta.json
  _meta/
    cik_ticker_map_edgar.csv
    cik_ticker_map.csv
```

Then run extraction (outside this doc) to produce:

- `*_item.json`
- `*_str.json`

These extracted JSON files are consumed by `vdbbuilder.py`.

## 2) Build vector database

Use `script/vdbbuilder.py` to:

- read extracted JSON (`*_item.json` or `*_str.json` based on scope),
- optionally filter by one or more years (`--year`),
- chunk text into units,
- generate embeddings,
- pool units into item vectors,
- compute orthogonal residuals,
- write parquet + npz + optional FAISS indices.

Outputs are scope-specific:

```text
<out_dir>/
  scope=all|heading|body/
    units/units_<YEAR>.parquet
    item_vectors/item_vectors_<YEAR>.parquet
    vectors/
      pooled/item=<ITEM>/year=<YEAR>.npz
      residual/item=<ITEM>/year=<YEAR>.npz
    indices/ ...                 # optional
```

## 3) Find peers

Use `script/peerfinder.py` to query peers for:

- focal firm (`--focalfirm`)
- year (`--year`)
- item(s) (`--item`)
- scope (`--scope`)
- method (`orthogonal`, `cosine`, `gemini`)

Modes:

- `orthogonal`: common-screen, specific-rank.
- `cosine`: direct pooled cosine ranking.
- `gemini`: text-to-text LLM scoring from `units.parquet`.

Precompute support:

- builds and caches NxN similarity tables for `orthogonal` and `cosine`.
- cache keys include `scope + item + year + method`.

## 4) Typical command sequence

```bash
# A. Download
python script/downloader.py --filing 10k --year 2024 --output_dir sec_filings --user_agent "YourApp/1.0 (you@example.com)"

# B. Build vectors (all text)
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --year 2024 --scope all --embedder local

# C. Query peers
python script/peerfinder.py --vdb_dir vector_db --scope all --focalfirm 0000320193 --year 2024 --item 1A --method orthogonal --k 20
```

## 5) Scope best practices

- Build each scope separately (`all`, `heading`, `body`).
- Query with matching scope in peerfinder.
- Do not mix scope outputs when comparing results.

## 6) Operational notes

- `local` embedding is deterministic and API-free.
- `openai` requires `OPENAI_API_KEY`.
- `gemini` method in peerfinder requires `GEMINI_API_KEY` (or `--gemini-api-key`).
- FAISS GPU is auto-used if bindings and GPU are available.
