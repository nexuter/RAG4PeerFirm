# Quickstart (First-Time Users)

This guide runs the pipeline end-to-end with the fewest decisions.

## 0) Prerequisites

- Python 3.10+
- SEC filing data already in `sec_filings/` with extracted JSON (`*_item.json`, `*_str.json`)
- Dependencies installed:

```bash
pip install -r requirements.txt
```

## 1) Build vector DB (recommended first run)

Use local embeddings (offline, deterministic) and `all` scope:

```bash
python script/vdbbuilder.py \
  --filing_dir sec_filings \
  --out_dir vector_db \
  --filing 10-K \
  --year 2024 \
  --scope all \
  --embedder local
```

Expected output:

```text
vector_db/scope=all/units.parquet
vector_db/scope=all/item_vectors.parquet
vector_db/scope=all/vectors/pooled/...
vector_db/scope=all/vectors/residual/...
```

## 2) Run first peer query (orthogonal method)

```bash
python script/peerfinder.py \
  --vdb_dir vector_db \
  --scope all \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --method orthogonal \
  --k 20 \
  --q_share 0.20 \
  --out_path output/peer_sets.csv.gz
```

## 3) Speed up repeated queries with precompute

```bash
python script/peerfinder.py \
  --vdb_dir vector_db \
  --scope all \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --method orthogonal \
  --precompute \
  --k 20
```

After this, future runs for the same `scope/item/year/method` read the cache first.

## 4) Compare other scopes (heading/body)

Build heading-only vectors:

```bash
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --year 2024 --scope heading --embedder local
```

Query heading-only peers:

```bash
python script/peerfinder.py --vdb_dir vector_db --scope heading --focalfirm 0000320193 --year 2024 --item 1A --method orthogonal --k 20
```

Repeat with `--scope body` to compare body-only behavior.

## 5) Optional: Gemini method

```bash
set GEMINI_API_KEY=YOUR_KEY
python script/peerfinder.py \
  --vdb_dir vector_db \
  --scope all \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --method gemini \
  --k 10
```

Default free-tier pacing is already configured:

- RPM `5`
- TPM `250000`
- RPD `20`

## Common first-run issues

1. `No source JSON filings found...`
- Check `--filing_dir` and extracted JSON files exist.

2. `No source JSON filings found for selected year(s)...`
- Check `--year` values exist in `sec_filings`.
- Remove `--year` to process all available years.

3. `No units/items built`
- Dataset may be too small for token thresholds. Lower:
  - `--min_unit_tokens`
  - `--min_units_per_item`

4. `No peers produced`
- Verify focal firm/year/item exists in built scope output.
- Ensure `--scope` in peerfinder matches scope used in vdbbuilder.

---

For deeper details, see:

- `docs/downloader.md`
- `docs/vdbbuilder.md`
- `docs/peerfinder.md`
- `docs/pipeline_walkthrough.md`
