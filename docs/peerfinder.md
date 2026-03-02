# Peer Finder Module Technical Guide

Module file: `script/peerfinder.py`

## Purpose

`peerfinder.py` returns item-level peer firms for a focal firm-year query using one of three methods:

- `orthogonal`: common-screen then specific-rank (residual-aware),
- `cosine`: direct pairwise cosine on pooled vectors,
- `llama3`: LLM-based text comparison using unit text.

It supports scope-aware querying (`all`, `heading`, `body`) and optional precomputed NxN matrices.

## Inputs and required artifacts

From `vdbbuilder` output:

```text
<vdb_dir>/scope=<scope>/item_vectors/item_vectors_<YEAR>.parquet
<vdb_dir>/scope=<scope>/vectors/pooled/item=<ITEM>/year=<YEAR>.npz
<vdb_dir>/scope=<scope>/vectors/residual/item=<ITEM>/year=<YEAR>.npz   # orthogonal method
<vdb_dir>/scope=<scope>/units/units_<YEAR>.parquet                      # llama3 method
```

If `scope=<scope>` folder does not exist, module falls back to `<vdb_dir>` for backward compatibility.

## Query model

Inputs:

- focal firm (`--focalfirm`)
- fiscal year (`--year`)
- one or more item ids (`--item`)
- method (`--method`)
- scope (`--scope`)

Output:

- one output row per selected peer, with `rank`, similarity columns, method, and source.

## Method details

## 1) `orthogonal`

For each requested item:

1. Load pooled matrix and residual matrix.
2. Candidate screening:
   - use pooled similarity (`q_share`) to select candidate set.
3. Ranking:
   - if focal residual is valid and candidates with valid residual exist:
     - rank by residual similarity (specific similarity).
   - otherwise:
     - fallback rank by pooled similarity.

Output fields:

- `sim_common`: pooled similarity (when computed in on-demand branch)
- `sim_specific`: residual similarity when available; `NaN` otherwise

### Orthogonal decomposition: intuition and math

This method tries to separate two signals inside each firm-item embedding:

1. **Common component**: language shared by many firms in the same item/year
   (industry boilerplate, regulatory wording, standard disclosures).
2. **Specific component**: language that is more firm-distinctive.

Let:

- \(E_{i,t,k} \in \mathbb{R}^d\): pooled item embedding for firm \(i\), year \(t\), item \(k\), L2-normalized.
- \(N\): number of firms available for \((t,k)\).

Construct leave-one-out common direction:

\[
\bar{E}_{-i,t,k} = \frac{1}{N-1} \sum_{j \neq i} E_{j,t,k}, \quad
\tilde{E}_{-i,t,k} = \frac{\bar{E}_{-i,t,k}}{\|\bar{E}_{-i,t,k}\|}
\]

Project focal vector onto this direction:

\[
\alpha_{i,t,k} = \langle E_{i,t,k}, \tilde{E}_{-i,t,k} \rangle
\]

Residual (orthogonal) part:

\[
E^\perp_{i,t,k} = E_{i,t,k} - \alpha_{i,t,k}\tilde{E}_{-i,t,k}
\]

If \(\|E^\perp_{i,t,k}\|\) is below threshold, the item is treated as mostly boilerplate (residual missing).
Otherwise normalize:

\[
\tilde{E}^\perp_{i,t,k} = \frac{E^\perp_{i,t,k}}{\|E^\perp_{i,t,k}\|}
\]

Peer search then uses two stages:

1. **Common-screen**: keep top `q_share` candidates by pooled cosine.
2. **Specific-rank**: among candidates, rank by residual cosine
   \(\cos(\tilde{E}^\perp_i, \tilde{E}^\perp_j)\) when both residuals are valid.
   Fallback to pooled cosine if residual is unavailable.

### Why this can outperform plain cosine

Pairwise cosine on pooled vectors:

\[
\cos(E_i, E_j)
\]

is simple and fast, but often rewards shared boilerplate heavily.

Orthogonal method reduces that effect by removing the dominant common direction before final ranking.
So it usually yields peers that are:

- less driven by generic item wording,
- more driven by firm-specific disclosure style/content.

### Expected outcomes in practice

- `sim_common` tends to be high for many firms in boilerplate-heavy items.
- residual norms are small when disclosures are near-generic; those firms/items may fallback to pooled ranking.
- when residual norms are healthy, top peers are often more differentiated than cosine-only top peers.

### When to prefer each method

- Prefer `orthogonal` when:
  - you care about firm-specific textual signal,
  - item language is known to be boilerplate-heavy.

- Prefer `cosine` when:
  - you need maximum speed/simplicity,
  - you want a baseline similarity metric,
  - residual availability is sparse in your dataset.

### Cost tradeoff vs cosine

- `cosine`: single-stage nearest-neighbor, cheapest.
- `orthogonal`: candidate screening + residual ranking, slightly more compute but usually still efficient with FAISS and/or precompute.

## 2) `cosine`

For each requested item:

1. Search pooled vectors by inner-product (equivalent to cosine for normalized vectors).
2. Exclude focal firm itself.
3. Return top `k`.

Output fields:

- `sim_common`: cosine score
- `sim_specific`: `NaN`

## 3) `llama3`

For each requested item:

1. Build text per firm by joining unit text from `units/units_<YEAR>.parquet`.
2. Send focal text + peer text to llama endpoint prompt.
3. Parse JSON response:
   - `score` in `[0,1]`
   - short `reason`
4. Rank peers by score descending and return top `k`.

Output fields:

- `sim_specific`: llama3 score
- `llama_reason`: short rationale
- `sim_common`: `NaN`

## Precompute cache

Supported methods: `orthogonal`, `cosine`.

Not supported: `llama3` (always on-demand).

Cache path:

```text
<active_vdb_dir>/precomputed/scope=<scope>/item=<ITEM>/year=<YEAR>/method=<METHOD>/
  similarity.npy
  firm_ids.json
```

Runtime behavior:

1. If cache exists: load and serve query from matrix row.
2. If cache missing:
   - with `--precompute`: build then use it,
   - without `--precompute`: compute on demand.
3. With `--precompute-overwrite`: rebuild even if cache exists.

## Llama3 runtime notes

- Llama3 runs on-demand only (no precompute path).
- Keep `--llama-max-chars` reasonable to control latency.
- Endpoint must be OpenAI-compatible (`/v1/chat/completions`).

## CLI reference

Core query:

- `--vdb_dir` (required)
- `--scope` (`all|heading|body`, default `all`)
- `--focalfirm` (required)
- `--year` (required)
- `--item` (required, one or more values)
- `--k` top peers (default `20`)

Method:

- `--method` (`orthogonal|cosine|llama3`, default `orthogonal`)
- `--q_share` candidate share for orthogonal screening (default `0.20`)

Precompute:

- `--precompute`
- `--precompute-overwrite`

FAISS:

- `--faiss-gpu` / `--no-faiss-gpu`

Llama3:

- `--llama-base-url` default `http://localhost:8321/v1`
- `--llama-api-key` optional (or env `LLAMA_API_KEY` / `OPENAI_API_KEY`)
- `--llama-model` default `llama3.3-70b`
- `--llama-max-chars` default `12000`
- `--llama-timeout-sec` default `120`

Output:

- `--out_path` CSV/CSV.GZ/Parquet path

## Output schema

Common columns:

- `focal_firm`
- `year`
- `item_id`
- `scope`
- `peer_firm`
- `rank`
- `sim_common`
- `sim_specific`
- `q_share`
- `k`
- `method`
- `source` (`precomputed` or `on_demand`)

Llama3-only:

- `llama_reason`

## Example commands

Orthogonal, on demand:

```bash
python script/peerfinder.py --vdb_dir vector_db --scope all --focalfirm 0000320193 --year 2024 --item 1A --method orthogonal --k 20 --q_share 0.2
```

Cosine with precompute:

```bash
python script/peerfinder.py --vdb_dir vector_db --scope body --focalfirm 0000320193 --year 2024 --item 1A --method cosine --precompute --k 20
```

Llama3:

```bash
python script/peerfinder.py --vdb_dir vector_db --scope heading --focalfirm 0000320193 --year 2024 --item 1A --method llama3 --llama-base-url http://localhost:8321/v1 --llama-model llama3.3-70b --k 10
```

## Troubleshooting

1. `No peers produced`
- Check scope matches built artifacts.
- Verify item/year/focal firm exists in vectors.
- For orthogonal, ensure residual file exists and contains candidates.

2. Llama3 returns many NaN scores
- Responses may be non-JSON from the endpoint.
- Verify endpoint/model and reduce run size.
- Lower `--llama-max-chars` to reduce prompt size.

3. Slow runtime
- Prefer precompute for `orthogonal` and `cosine`.
- For llama3, limit peers/items or run in batches.
