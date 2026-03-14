"""
peerfinder.py

Find peer firms with a two-stage workflow:
1. cosine similarity over one pooled vector per `(firm, year, item)` to keep the top `q_share`
2. local LLM reranking over retrieved item text to produce the final top peers

Scope choices:
- `all`: raw item text from `*_item.json`
- `heading`: heading-only text from `*_str.json`
- `summary`: local summary text from `*_item_summ.json`
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import faiss  # type: ignore
except Exception as exc:
    raise RuntimeError("faiss is required for peerfinder. Install faiss-cpu or faiss-gpu.") from exc


def normalize_item_id(item: str) -> str:
    return re.sub(r"\s+", "", item.strip().upper())


def _resolve_scope_dir(vdb_dir: Path, scope: str) -> Path:
    scoped = vdb_dir / f"scope={scope}"
    return scoped if scoped.exists() else vdb_dir


def load_pooled(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["mat"].astype(np.float32), data["ids"].astype(str)


def load_item_vectors(vdb_dir: Path, year: int, scope: str) -> pd.DataFrame:
    scope_dir = _resolve_scope_dir(vdb_dir, scope)
    path = scope_dir / "item_vectors" / f"item_vectors_{int(year)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing item vector table: {path}")
    return pd.read_parquet(path)


def load_faiss_index(vdb_dir: Path, scope: str, item_id: str, year: int) -> Tuple["faiss.Index", np.ndarray]:
    scope_dir = _resolve_scope_dir(vdb_dir, scope)
    idx_dir = scope_dir / "indices" / f"item={item_id}" / f"year={year}"
    index = faiss.read_index(str(idx_dir / "pooled.faiss"))
    ids = np.array(json.loads((idx_dir / "pooled_ids.json").read_text(encoding="utf-8")), dtype=str)
    return index, ids


def get_similarity_candidates(
    vdb_dir: Path,
    scope: str,
    item_id: str,
    year: int,
    focal_firm: str,
    q_share: float,
) -> Tuple[List[Tuple[str, float]], int]:
    scope_dir = _resolve_scope_dir(vdb_dir, scope)
    pooled_path = scope_dir / "vectors" / "pooled" / f"item={item_id}" / f"year={year}.npz"
    mat, firm_ids = load_pooled(pooled_path)
    matches = np.where(firm_ids == str(focal_firm))[0]
    if matches.size == 0:
        raise RuntimeError(f"Focal firm {focal_firm} not found for item={item_id}, year={year}")
    focal_idx = int(matches[0])
    n = int(mat.shape[0])
    k = min(n, max(2, int(math.ceil(n * q_share)) + 1))

    try:
        index, idx_ids = load_faiss_index(vdb_dir, scope, item_id, year)
        if list(idx_ids) != list(firm_ids):
            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat.astype(np.float32))
    except Exception:
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat.astype(np.float32))

    scores, ids = index.search(mat[focal_idx].reshape(1, -1).astype(np.float32), k)
    rows: List[Tuple[str, float]] = []
    for idx, score in zip(ids[0], scores[0]):
        if int(idx) < 0:
            continue
        peer_id = str(firm_ids[int(idx)])
        if peer_id == str(focal_firm):
            continue
        rows.append((peer_id, float(score)))
    return rows, n


def _collect_node_values(node: object, field: str, out: List[str]) -> None:
    if isinstance(node, dict):
        value = node.get(field)
        if isinstance(value, str):
            value = value.strip()
            if value:
                out.append(value)
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                _collect_node_values(child, field, out)
    elif isinstance(node, list):
        for child in node:
            _collect_node_values(child, field, out)


def load_item_text(source_path: str, item_id: str, scope: str) -> str:
    path = Path(source_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    item_id = normalize_item_id(item_id)

    if scope == "summary":
        items = payload.get("items", {})
        value = items.get(item_id) or items.get(str(item_id))
        if isinstance(value, dict):
            return str(value.get("summary") or "").strip()
        return ""

    if scope == "all":
        items = payload.get("items", {})
        value = items.get(item_id) or items.get(str(item_id))
        if isinstance(value, dict):
            return str(value.get("text_content") or value.get("html_content") or "").strip()
        return ""

    structures = payload.get("structures", {})
    value = structures.get(item_id) or structures.get(str(item_id))
    if value is None:
        by_norm = {normalize_item_id(str(k)): v for k, v in structures.items()} if isinstance(structures, dict) else {}
        value = by_norm.get(item_id)
    if value is None:
        return ""
    parts: List[str] = []
    _collect_node_values(value, "heading", parts)
    return "\n\n".join(parts)


def truncate_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


def rerank_prompt(item_id: str, scope: str, focal_text: str, peer_text: str) -> str:
    return f"""You are ranking whether two firms are close peers for SEC filing item {item_id}.

Use only the provided {scope} text. Do not rely on outside knowledge.
Reward overlap in:
- business model, products, end markets, customer type, and geography
- strategic priorities and recent strategic change
- risk profile, regulation, supply chain, and operating constraints
- capital structure, allocation, or financial posture when clearly stated

Penalize matches driven only by generic SEC language.

Return strict JSON only:
{{"score": <integer 0-100>, "reason": "<one short sentence>"}}

Firm A text:
{focal_text}

Firm B text:
{peer_text}
"""


def ollama_rerank(
    base_url: str,
    model: str,
    prompt: str,
    timeout_sec: int,
) -> Tuple[float, str]:
    resp = requests.post(
        base_url.rstrip("/") + "/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    text = str(resp.json().get("response") or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Reranker returned non-JSON payload: {text[:200]}")
        payload = json.loads(match.group(0))
    score = float(payload.get("score", 0.0))
    reason = str(payload.get("reason") or "").strip()
    score = max(0.0, min(100.0, score))
    return score / 100.0, reason


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def run_peerfinder(
    vdb_dir: Path,
    focal_firm: str,
    year: int,
    items: Sequence[str],
    scope: str,
    q_share: float,
    top_share: float,
    model: str,
    timeout_sec: int,
    ollama_url: str,
    max_chars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    items_df = load_item_vectors(vdb_dir, year, scope)
    item_set = {normalize_item_id(x) for x in items}
    detail_rows: List[Dict[str, object]] = []
    universe_sizes: List[int] = []

    for item_id in item_set:
        sub = items_df[(items_df["item_id"] == item_id) & (items_df["year"] == int(year))]
        if sub.empty:
            continue
        focal_row = sub[sub["firm_id"].astype(str) == str(focal_firm)]
        if focal_row.empty:
            continue

        candidates, universe_n = get_similarity_candidates(vdb_dir, scope, item_id, year, focal_firm, q_share)
        universe_sizes.append(universe_n)
        if not candidates:
            continue

        source_by_firm = {
            str(row["firm_id"]): str(row["source_path"])
            for _, row in sub[["firm_id", "source_path"]].drop_duplicates().iterrows()
        }
        focal_text = truncate_text(load_item_text(source_by_firm[str(focal_firm)], item_id, scope), max_chars)
        if not focal_text:
            continue

        for peer_id, cosine_score in candidates:
            peer_path = source_by_firm.get(str(peer_id))
            if not peer_path:
                continue
            peer_text = truncate_text(load_item_text(peer_path, item_id, scope), max_chars)
            if not peer_text:
                continue
            rerank_score, reason = ollama_rerank(
                base_url=ollama_url,
                model=model,
                prompt=rerank_prompt(item_id, scope, focal_text, peer_text),
                timeout_sec=timeout_sec,
            )
            combined = 0.3 * float(cosine_score) + 0.7 * rerank_score
            detail_rows.append(
                {
                    "focal_firm": str(focal_firm),
                    "year": int(year),
                    "item_id": item_id,
                    "peer_firm": str(peer_id),
                    "cosine_score": float(cosine_score),
                    "rerank_score": float(rerank_score),
                    "combined_item_score": float(combined),
                    "scope": scope,
                    "model": model,
                    "reason": reason,
                }
            )

    if not detail_rows:
        raise RuntimeError("No peer candidates were produced. Check inputs and available vectors.")

    detail_df = pd.DataFrame(detail_rows)
    requested_items = max(1, len(item_set))
    agg = (
        detail_df.groupby(["focal_firm", "year", "peer_firm"], as_index=False)
        .agg(
            items_matched=("item_id", "nunique"),
            avg_cosine=("cosine_score", "mean"),
            avg_rerank=("rerank_score", "mean"),
            avg_item_score=("combined_item_score", "mean"),
        )
    )
    agg["coverage_share"] = agg["items_matched"] / requested_items
    agg["final_score"] = agg["avg_item_score"] * agg["coverage_share"]
    agg = agg.sort_values(["final_score", "avg_rerank", "avg_cosine"], ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    universe_n = max(universe_sizes) if universe_sizes else len(agg) + 1
    final_k = max(1, int(math.ceil(universe_n * top_share)))
    final_df = agg.head(final_k).copy()
    return final_df, detail_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Find top peer firms by cosine retrieval plus local LLM reranking")
    parser.add_argument("--vdb_dir", required=True, help="Vector DB directory from vdbbuilder.py")
    parser.add_argument("--scope", default="all", choices=["heading", "all", "summary"])
    parser.add_argument("--focalfirm", required=True, help="Focal firm id")
    parser.add_argument("--year", required=True, type=int, help="Query year")
    parser.add_argument("--item", nargs="+", required=True, help="One or more item ids")
    parser.add_argument("--q_share", type=float, default=0.10, help="Cosine candidate share before reranking")
    parser.add_argument("--top_share", type=float, default=0.01, help="Final share kept after reranking")
    parser.add_argument("--model", default="llama3.2:8b", help="Local Ollama reranker model")
    parser.add_argument("--timeout", type=int, default=300, help="LLM timeout in seconds")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL for local Ollama")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max chars of item text sent to the LLM")
    parser.add_argument(
        "--out_path",
        default="output/peer_sets_{timestamp}.csv",
        help="Output path template for final ranking table. Supports {timestamp}.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(str(args.out_path).format(timestamp=timestamp))
    final_df, detail_df = run_peerfinder(
        vdb_dir=Path(args.vdb_dir),
        focal_firm=str(args.focalfirm),
        year=int(args.year),
        items=[normalize_item_id(x) for x in args.item],
        scope=str(args.scope),
        q_share=float(args.q_share),
        top_share=float(args.top_share),
        model=str(args.model),
        timeout_sec=int(args.timeout),
        ollama_url=str(args.ollama_url),
        max_chars=int(args.max_chars),
    )
    write_table(final_df, out_path)
    detail_path = out_path.with_name(out_path.stem + "_detail" + out_path.suffix)
    write_table(detail_df, detail_path)
    print(f"[DONE] Wrote final ranking: {out_path}")
    print(f"[DONE] Wrote detailed rerank rows: {detail_path}")


if __name__ == "__main__":
    main()
