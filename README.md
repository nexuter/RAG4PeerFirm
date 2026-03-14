# RAG4PeerFirm

Local pipeline for SEC item extraction, item summarization, vector building, peer-firm retrieval, LLM reranking, and evaluation.

## Workflow

1. Extract filing items into `*_item.json` and `*_str.json`.
2. Run `script/summrizer.py` to create sibling `*_item_summ.json` summaries.
3. Run `script/vdbbuilder.py` for `all`, `heading`, or `summary` scope.
4. Run `script/peerfinder.py` to retrieve top 10% by cosine and rerank to top 1%.
5. Run `script/evaluate.py` with optional metadata and analyst peer labels.

## Scripts

- `script/summrizer.py`
  - Local Ollama summarizer with map-reduce and previous-year context.
- `script/vdbbuilder.py`
  - Local embedding build with FAISS indices.
  - One pooled vector per `(firm, year, item)` for every scope.
- `script/peerfinder.py`
  - Cosine retrieval plus local LLM reranking on retrieved item text.
- `script/evaluate.py`
  - Weak-label evaluation with `Recall@50`, `NDCG@10`, and year-over-year stability.

## Install

```bash
pip install -r requirements.txt
```

## Example

Summarize:

```bash
python script/summrizer.py --filing_dir sec_filings --filing 10-K --year 2024 --llm llama3.2:3b
```

Build vectors:

```bash
python script/vdbbuilder.py --filing_dir sec_filings --out_dir vector_db --filing 10-K --year 2024 --scope summary
```

Find peers:

```bash
python script/peerfinder.py \
  --vdb_dir vector_db \
  --scope summary \
  --focalfirm 0000320193 \
  --year 2024 \
  --item 1A 7 7A \
  --model llama3.2:8b
```

Evaluate:

```bash
python script/evaluate.py --peer_path output/peer_sets_20240101_120000.csv --metadata_path firm_metadata.csv
```
