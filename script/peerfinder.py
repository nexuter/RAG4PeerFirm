"""
Peer finder for item-by-item matching using orthogonal projection outputs from vdbbuilder.

Procedure per item/year:
1) Candidate screen by pooled/common similarity.
2) Final rank by residual/specific similarity (fallback to pooled if residual unavailable).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as exc:
    raise RuntimeError("faiss is required for peerfinder. Install faiss-cpu or faiss-gpu.") from exc


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_pooled(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["mat"].astype(np.float32), data["ids"].astype(str)


def load_residual(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["mat"].astype(np.float32), data["ids"].astype(str), data["mask"].astype(np.int8)


def build_index(mat: np.ndarray) -> "faiss.Index":
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat.astype(np.float32))
    return idx


def faiss_topk(index: "faiss.Index", query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims, ids = index.search(query.reshape(1, -1).astype(np.float32), k)
    return ids[0], sims[0]


def item_candidates(
    pooled_mat: np.ndarray,
    focal_idx: int,
    q_share: float,
) -> np.ndarray:
    n = pooled_mat.shape[0]
    k = max(2, int(math.ceil(n * q_share)) + 1)
    idx = build_index(pooled_mat)
    cand_ids, _ = faiss_topk(idx, pooled_mat[focal_idx], k)
    cand_ids = cand_ids[cand_ids >= 0]
    cand_ids = cand_ids[cand_ids != focal_idx]
    return cand_ids


def load_item_vectors(vdb_dir: Path) -> pd.DataFrame:
    parquet_path = vdb_dir / "item_vectors.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Missing {parquet_path}")


def run_peerfinder(
    *,
    vdb_dir: Path,
    focalfirm: str,
    year: int,
    item: str,
    top_k: int,
    q_share: float,
) -> pd.DataFrame:
    item_vectors = load_item_vectors(vdb_dir)
    if item.lower() == "all":
        items = sorted(item_vectors[item_vectors["year"] == year]["item_id"].dropna().astype(str).unique().tolist())
    else:
        items = [item.upper()]

    rows: List[Dict[str, object]] = []

    for item_id in items:
        pooled_path = vdb_dir / "vectors" / "pooled" / f"item={item_id}" / f"year={year}.npz"
        residual_path = vdb_dir / "vectors" / "residual" / f"item={item_id}" / f"year={year}.npz"
        if not pooled_path.exists() or not residual_path.exists():
            continue

        pooled_mat, firm_ids = load_pooled(pooled_path)
        residual_mat, residual_ids, residual_mask = load_residual(residual_path)
        if not np.array_equal(firm_ids, residual_ids):
            raise RuntimeError(f"ID alignment mismatch for item={item_id}, year={year}")

        where = np.where(firm_ids == str(focalfirm))[0]
        if len(where) == 0:
            continue
        focal_idx = int(where[0])

        candidates = item_candidates(pooled_mat, focal_idx, q_share=q_share)
        if len(candidates) == 0:
            continue

        focal_has_residual = bool(residual_mask[focal_idx] == 1)
        if focal_has_residual:
            valid_candidates = candidates[residual_mask[candidates].astype(bool)]
            if len(valid_candidates) > 0:
                cand_mat = residual_mat[valid_candidates]
                idx = build_index(cand_mat)
                k_eff = min(top_k, len(valid_candidates))
                local_ids, spec_sims = faiss_topk(idx, residual_mat[focal_idx], k_eff)
                local_ids = local_ids[local_ids >= 0]
                peer_idx = valid_candidates[local_ids]
                common_sims = pooled_mat[peer_idx] @ pooled_mat[focal_idx]
                for rank, j in enumerate(peer_idx, start=1):
                    rows.append(
                        {
                            "focal_firm": str(focalfirm),
                            "year": int(year),
                            "item_id": item_id,
                            "peer_firm": str(firm_ids[j]),
                            "rank": rank,
                            "sim_common": float(common_sims[rank - 1]),
                            "sim_specific": float(spec_sims[rank - 1]),
                            "q_share": float(q_share),
                            "k": int(top_k),
                        }
                    )
                continue

        cand_mat = pooled_mat[candidates]
        idx = build_index(cand_mat)
        k_eff = min(top_k, len(candidates))
        local_ids, common_sims = faiss_topk(idx, pooled_mat[focal_idx], k_eff)
        local_ids = local_ids[local_ids >= 0]
        peer_idx = candidates[local_ids]
        for rank, j in enumerate(peer_idx, start=1):
            rows.append(
                {
                    "focal_firm": str(focalfirm),
                    "year": int(year),
                    "item_id": item_id,
                    "peer_firm": str(firm_ids[j]),
                    "rank": rank,
                    "sim_common": float(common_sims[rank - 1]),
                    "sim_specific": np.nan,
                    "q_share": float(q_share),
                    "k": int(top_k),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No peers produced. Check focal firm/item/year availability in built vectors.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Find peers using common-screen and specific-rank")
    parser.add_argument("--vdb_dir", required=True, help="Output directory generated by vdbbuilder")
    parser.add_argument("--focalfirm", required=True, help="Focal firm id/CIK")
    parser.add_argument("--year", type=int, required=True, help="Fiscal year")
    parser.add_argument("--item", default="all", help="Item id (e.g., 1A) or 'all'")
    parser.add_argument("--k", type=int, default=20, help="Top peers per item")
    parser.add_argument("--q_share", type=float, default=0.20, help="Candidate screening share")
    parser.add_argument("--out_path", default="output/peer_sets.csv.gz", help="Output csv/parquet path")
    args = parser.parse_args()

    out_df = run_peerfinder(
        vdb_dir=Path(args.vdb_dir),
        focalfirm=str(args.focalfirm),
        year=int(args.year),
        item=str(args.item),
        top_k=int(args.k),
        q_share=float(args.q_share),
    )

    out_path = Path(args.out_path)
    ensure_dir(out_path)
    if out_path.suffix.lower() == ".parquet":
        out_df.to_parquet(out_path, index=False)
    else:
        if out_path.suffix.lower() not in {".csv", ".gz", ".csv.gz"}:
            out_path = out_path.with_suffix(".csv.gz")
        out_df.to_csv(out_path, index=False, compression="gzip")

    print(f"[DONE] Wrote peer sets: {out_path}")


if __name__ == "__main__":
    main()
