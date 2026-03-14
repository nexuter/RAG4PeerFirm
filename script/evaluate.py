"""
evaluate.py

Evaluate peer rankings with weak labels and save timestamped outputs under
 `./output` by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def band_match(a: object, b: object, ratio: float = 2.0) -> bool:
    try:
        x = float(a)
        y = float(b)
    except Exception:
        return False
    if x <= 0 or y <= 0:
        return False
    hi = max(x, y)
    lo = min(x, y)
    return hi / lo <= ratio


def build_label_sets(
    ranking_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    analyst_df: Optional[pd.DataFrame],
) -> Dict[Tuple[str, int], Set[str]]:
    labels: Dict[Tuple[str, int], Set[str]] = {}
    if metadata_df is not None and not metadata_df.empty:
        meta = metadata_df.copy()
        meta["firm_id"] = meta["firm_id"].astype(str)
        meta["year"] = meta["year"].astype(int)
        cols = set(meta.columns)
        required = {"firm_id", "year"}
        if required.issubset(cols):
            for (focal_firm, year), focal_rows in meta.groupby(["firm_id", "year"]):
                focal = focal_rows.iloc[0]
                peers = meta[meta["year"] == year].copy()
                mask = pd.Series(False, index=peers.index)
                for col in ("sic", "naics", "gics"):
                    if col in cols and pd.notna(focal.get(col)):
                        mask = mask | (peers[col].astype(str) == str(focal.get(col)))
                if "market_cap" in cols:
                    mask = mask | peers["market_cap"].apply(lambda x: band_match(x, focal.get("market_cap")))
                if "revenue" in cols:
                    mask = mask | peers["revenue"].apply(lambda x: band_match(x, focal.get("revenue")))
                peer_ids = set(peers.loc[mask, "firm_id"].astype(str).tolist())
                peer_ids.discard(str(focal_firm))
                labels[(str(focal_firm), int(year))] = peer_ids

    if analyst_df is not None and not analyst_df.empty:
        analyst = analyst_df.copy()
        analyst["focal_firm"] = analyst["focal_firm"].astype(str)
        analyst["peer_firm"] = analyst["peer_firm"].astype(str)
        analyst["year"] = analyst["year"].astype(int)
        for (focal_firm, year), sub in analyst.groupby(["focal_firm", "year"]):
            labels.setdefault((str(focal_firm), int(year)), set()).update(sub["peer_firm"].tolist())
    return labels


def recall_at_k(preds: Sequence[str], labels: Set[str], k: int) -> float:
    if not labels:
        return float("nan")
    top = preds[:k]
    return len(set(top) & labels) / max(1, len(labels))


def ndcg_at_k(preds: Sequence[str], labels: Set[str], k: int) -> float:
    if not labels:
        return float("nan")
    gains = [1.0 if peer in labels else 0.0 for peer in preds[:k]]
    dcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))
    ideal_hits = min(k, len(labels))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    if idcg == 0:
        return float("nan")
    return float(dcg / idcg)


def stability_across_years(ranking_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    df = ranking_df.copy()
    df["focal_firm"] = df["focal_firm"].astype(str)
    df["year"] = df["year"].astype(int)
    df["peer_firm"] = df["peer_firm"].astype(str)
    for focal_firm, sub in df.groupby("focal_firm"):
        years = sorted(sub["year"].unique().tolist())
        for prev_year, curr_year in zip(years, years[1:]):
            prev_set = set(sub[sub["year"] == prev_year].sort_values("rank")["peer_firm"].head(top_k).tolist())
            curr_set = set(sub[sub["year"] == curr_year].sort_values("rank")["peer_firm"].head(top_k).tolist())
            union = prev_set | curr_set
            jaccard = float(len(prev_set & curr_set) / len(union)) if union else float("nan")
            rows.append(
                {
                    "focal_firm": str(focal_firm),
                    "prev_year": int(prev_year),
                    "year": int(curr_year),
                    "top_k": int(top_k),
                    "jaccard": jaccard,
                }
            )
    return pd.DataFrame(rows)


def evaluate_rankings(
    ranking_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    analyst_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = build_label_sets(ranking_df, metadata_df, analyst_df)
    metric_rows: List[Dict[str, object]] = []

    df = ranking_df.copy()
    df["focal_firm"] = df["focal_firm"].astype(str)
    df["year"] = df["year"].astype(int)
    df["peer_firm"] = df["peer_firm"].astype(str)

    for (focal_firm, year), sub in df.groupby(["focal_firm", "year"]):
        ranked = sub.sort_values("rank")["peer_firm"].tolist()
        label_set = labels.get((str(focal_firm), int(year)), set())
        metric_rows.append(
            {
                "focal_firm": str(focal_firm),
                "year": int(year),
                "label_count": int(len(label_set)),
                "recall_at_50": recall_at_k(ranked, label_set, 50),
                "ndcg_at_10": ndcg_at_k(ranked, label_set, 10),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    stability_df = stability_across_years(df, top_k=10)
    return metrics_df, stability_df


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate peer-firm ranking outputs")
    parser.add_argument("--peer_path", required=True, help="Ranking file from peerfinder.py")
    parser.add_argument("--metadata_path", default=None, help="Optional firm metadata CSV/Parquet")
    parser.add_argument("--analyst_peers_path", default=None, help="Optional analyst peer-set CSV/Parquet")
    parser.add_argument("--out_dir", default="output", help="Directory for timestamped evaluation outputs")
    args = parser.parse_args()

    ranking_path = Path(args.peer_path)
    ranking_df = load_table(ranking_path)
    metadata_df = load_table(Path(args.metadata_path)) if args.metadata_path else None
    analyst_df = load_table(Path(args.analyst_peers_path)) if args.analyst_peers_path else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking_copy = out_dir / f"peer_ranking_{timestamp}{ranking_path.suffix or '.csv'}"
    write_table(ranking_df, ranking_copy)

    metrics_df, stability_df = evaluate_rankings(ranking_df, metadata_df, analyst_df)
    metrics_path = out_dir / f"peer_eval_metrics_{timestamp}.csv"
    stability_path = out_dir / f"peer_eval_stability_{timestamp}.csv"
    write_table(metrics_df, metrics_path)
    write_table(stability_df, stability_path)

    summary = {
        "peer_path": str(ranking_path),
        "ranking_copy": str(ranking_copy),
        "metrics_path": str(metrics_path),
        "stability_path": str(stability_path),
        "num_rank_rows": int(len(ranking_df)),
        "num_metric_rows": int(len(metrics_df)),
        "num_stability_rows": int(len(stability_df)),
    }
    summary_path = out_dir / f"peer_eval_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote ranking copy: {ranking_copy}")
    print(f"[DONE] Wrote metrics: {metrics_path}")
    print(f"[DONE] Wrote stability: {stability_path}")
    print(f"[DONE] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
