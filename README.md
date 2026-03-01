# RAG4PeerFirm

SEC filing pipeline for download, extraction, vector-db build, and peer-firm matching.

## Project Layout

- `script/downloader.py`: download filings from SEC EDGAR (supports ticker/CIK filters)
- `script/extractor.py`: extract filing items and structure JSON from downloaded filings
- `script/vdbbuilder.py`: build unit/item vectors, residual vectors, and FAISS indices
- `script/peerfinder.py`: run common-screen + specific-rank peer matching
- `utils/downloader.py`: SEC downloader client
- `utils/index_parser.py`: SEC full-index parser
- `utils/file_manager.py`: file IO helpers
- `utils/parser.py`, `utils/extractor.py`, `utils/structure_extractor.py`: extraction internals
- `utils/config.py`: shared constants
- `tests/smoke_test.py`: lightweight compile/import smoke test

## Install

```bash
pip install -r requirements.txt
```

## Smoke Test

```bash
python tests/smoke_test.py
```

## 1) Download Filings

```bash
python script/downloader.py \
  --filing 10k \
  --year 2024 \
  --output_dir sec_filings \
  --user_agent "RAG4PeerFirm/1.0 (your-email@example.com)"
```

Examples:

```bash
# specific tickers
python script/downloader.py --filing 10k --year 2024 --ticker AAPL MSFT --output_dir sec_filings --user_agent "RAG4PeerFirm/1.0 (your-email@example.com)"

# specific CIKs
python script/downloader.py --filing 10k --year 2024 --cik 0000320193 0000789019 --output_dir sec_filings --user_agent "RAG4PeerFirm/1.0 (your-email@example.com)"
```

## 2) Extract Items and Structure

```bash
python script/extractor.py \
  --filing_dir sec_filings \
  --filing 10-K \
  --year 2024
```

## 3) Build Vector DB

```bash
python script/vdbbuilder.py \
  --filing_dir sec_filings \
  --out_dir vector_db \
  --filing 10-K \
  --embedder local
```

OpenAI embeddings:

```bash
set OPENAI_API_KEY=...
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --embedder openai
```

Outputs under `<out_dir>`:

- `units.parquet`
- `item_vectors.parquet`
- `vectors/pooled/item=*/year=*.npz`
- `vectors/residual/item=*/year=*.npz`
- `indices/item=*/year=*/pooled.faiss` (+ residual where available)

## 4) Find Peers

```bash
python script/peerfinder.py \
  --vdb_dir vector_db \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A \
  --k 20 \
  --q_share 0.20 \
  --out_path output/peer_sets.csv.gz
```

Use `--item all` to run across all available items.

## Notes

- `peerfinder` requires FAISS.
- `vdbbuilder --embedder local` is deterministic and works without external API calls.
- Downloader writes run artifacts under `logs/` and `stats/`.
