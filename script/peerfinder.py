"""
peerfinder.py

Find item-level peer firms from artifacts produced by `vdbbuilder.py`.

Inputs:
- Built vector DB directory (`--vdb_dir`) containing pooled/residual vectors by item/year.
- Query tuple: focal firm (`--focalfirm`), year (`--year`), and one or more items (`--item`).
- Scope (`--scope`): `all`, `heading`, or `body`.
- Similarity mode (`--method`):
  - `orthogonal`: common-screen then specific-rank behavior.
  - `cosine`: direct pairwise cosine on pooled item embeddings.
  - `llama3`: local LLM comparison on item text via llama-stack endpoint.

Design:
1. Load per-item-year pooled and residual matrices.
2. If precomputed matrix exists, use it directly.
3. Else compute on demand (or build cache when `--precompute` is set for non-LLM methods).
4. Return top-k peers per requested item.

Precompute cache:
- Path: `precomputed/item=<ITEM>/year=<YEAR>/method=<METHOD>/`
- Files:
  - `similarity.npy` (NxN similarity matrix)
  - `firm_ids.json` (row/column id alignment)

Output:
- CSV/Parquet table with peer rows and a `source` column (`precomputed` or `on_demand`).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as exc:
    raise RuntimeError("faiss is required for peerfinder. Install faiss-cpu or faiss-gpu.") from exc


def ensure_dir(path: Path) -> None:
    """Create output parent directory when missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_pooled(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load pooled vectors and aligned firm ids from npz."""
    try:
        data = np.load(path, allow_pickle=False)
        return data["mat"].astype(np.float32), data["ids"].astype(str)
    except ValueError as exc:
        # Backward compatibility: older builds may have object-dtype ids in NPZ.
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        data = np.load(path, allow_pickle=True)
        return data["mat"].astype(np.float32), data["ids"].astype(str)


def load_residual(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load residual vectors, aligned firm ids, and validity mask from npz."""
    try:
        data = np.load(path, allow_pickle=False)
        return data["mat"].astype(np.float32), data["ids"].astype(str), data["mask"].astype(np.int8)
    except ValueError as exc:
        # Backward compatibility: older builds may have object-dtype ids in NPZ.
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        data = np.load(path, allow_pickle=True)
        return data["mat"].astype(np.float32), data["ids"].astype(str), data["mask"].astype(np.int8)


def _faiss_gpu_available() -> bool:
    required = ("get_num_gpus", "index_cpu_to_all_gpus")
    if not all(hasattr(faiss, name) for name in required):
        return False
    try:
        return int(faiss.get_num_gpus()) > 0
    except Exception:
        return False


def _system_gpu_present() -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return proc.returncode == 0 and "GPU " in (proc.stdout or "")
    except Exception:
        return False


def _warn_if_gpu_available_without_faiss_gpu(requested_gpu: bool) -> None:
    if not requested_gpu:
        return
    if _faiss_gpu_available():
        return
    if _system_gpu_present():
        print(
            "[WARN] NVIDIA GPU detected, but FAISS GPU bindings are unavailable. "
            "Install faiss-gpu to enable GPU acceleration. Continuing with CPU FAISS."
        )


def build_index(mat: np.ndarray, use_gpu: bool = True) -> "faiss.Index":
    """Build inner-product FAISS index on CPU or GPU (if available and enabled)."""
    cpu_idx = faiss.IndexFlatIP(mat.shape[1])
    vecs = mat.astype(np.float32)
    if use_gpu and _faiss_gpu_available():
        try:
            gpu_idx = faiss.index_cpu_to_all_gpus(cpu_idx)
            gpu_idx.add(vecs)
            return gpu_idx
        except Exception:
            pass
    cpu_idx.add(vecs)
    return cpu_idx


def _resolve_scope_dir(vdb_dir: Path, scope: str) -> Path:
    """Resolve scoped artifact directory (`vdb_dir/scope=<scope>`), with fallback to root."""
    scoped = vdb_dir / f"scope={scope}"
    return scoped if scoped.exists() else vdb_dir


def _precomputed_dir(vdb_dir: Path, scope: str, item_id: str, year: int, method: str) -> Path:
    """Return directory path for one item/year/method precomputed matrix."""
    return vdb_dir / "precomputed" / f"scope={scope}" / f"item={item_id}" / f"year={year}" / f"method={method}"


def _precomputed_paths(vdb_dir: Path, scope: str, item_id: str, year: int, method: str) -> Tuple[Path, Path]:
    """Return `(similarity_path, ids_path)` for cached matrix artifacts."""
    base = _precomputed_dir(vdb_dir, scope, item_id, year, method)
    return base / "similarity.npy", base / "firm_ids.json"


def _precomputed_exists(vdb_dir: Path, scope: str, item_id: str, year: int, method: str) -> bool:
    """Check whether both cached matrix and id map files exist."""
    mat_path, ids_path = _precomputed_paths(vdb_dir, scope, item_id, year, method)
    return mat_path.is_file() and ids_path.is_file()


def _compute_similarity_matrix(
    method: str,
    pooled_mat: np.ndarray,
    residual_mat: np.ndarray,
    residual_mask: np.ndarray,
) -> np.ndarray:
    """
    Build full NxN similarity matrix.

    - `cosine`: pooled cosine for all pairs.
    - `orthogonal`: pooled/common cosine by default, replaced with residual/specific cosine
      where both firms have valid residual vectors.
    """
    if method == "cosine":
        return (pooled_mat @ pooled_mat.T).astype(np.float32)

    # Orthogonal mode: use specific residual cosine where both firms are valid,
    # otherwise fallback to pooled/common cosine.
    common = (pooled_mat @ pooled_mat.T).astype(np.float32)
    valid = residual_mask.astype(bool)
    if not valid.any():
        return common

    specific = (residual_mat @ residual_mat.T).astype(np.float32)
    use_specific = np.outer(valid, valid)
    common[use_specific] = specific[use_specific]
    return common


def _save_precomputed(
    vdb_dir: Path,
    scope: str,
    item_id: str,
    year: int,
    method: str,
    matrix: np.ndarray,
    firm_ids: np.ndarray,
) -> None:
    """Persist precomputed NxN matrix and aligned firm id ordering."""
    mat_path, ids_path = _precomputed_paths(vdb_dir, scope, item_id, year, method)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mat_path, matrix.astype(np.float32))
    ids_path.write_text(json.dumps([str(x) for x in firm_ids.tolist()]), encoding="utf-8")


def _load_precomputed(vdb_dir: Path, scope: str, item_id: str, year: int, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed matrix and aligned ids."""
    mat_path, ids_path = _precomputed_paths(vdb_dir, scope, item_id, year, method)
    matrix = np.load(mat_path).astype(np.float32)
    ids = np.array(json.loads(ids_path.read_text(encoding="utf-8")), dtype=str)
    return matrix, ids


def _topk_from_similarity_row(scores: np.ndarray, self_idx: int, k: int) -> np.ndarray:
    """Return top-k peer indices from one similarity row, excluding focal self index."""
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    n = scores.shape[0]
    if n <= 1:
        return np.array([], dtype=np.int64)
    safe = scores.copy()
    safe[self_idx] = -np.inf
    k_eff = min(k, n - 1)
    idx = np.argpartition(-safe, k_eff - 1)[:k_eff]
    return idx[np.argsort(-safe[idx])]


def faiss_topk(index: "faiss.Index", query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index and return `(indices, similarities)` for one query vector."""
    sims, ids = index.search(query.reshape(1, -1).astype(np.float32), k)
    return ids[0], sims[0]


def item_candidates(
    pooled_mat: np.ndarray,
    focal_idx: int,
    q_share: float,
    use_gpu: bool,
) -> np.ndarray:
    """Screen candidate firms by pooled/common similarity top-q share."""
    n = pooled_mat.shape[0]
    k = max(2, int(math.ceil(n * q_share)) + 1)
    idx = build_index(pooled_mat, use_gpu=use_gpu)
    cand_ids, _ = faiss_topk(idx, pooled_mat[focal_idx], k)
    cand_ids = cand_ids[cand_ids >= 0]
    cand_ids = cand_ids[cand_ids != focal_idx]
    return cand_ids


def load_item_vectors(vdb_dir: Path, year: int) -> pd.DataFrame:
    """Load item-vector metadata for requested year (new per-year layout with legacy fallback)."""
    year_path = vdb_dir / "item_vectors" / f"item_vectors_{int(year)}.parquet"
    if year_path.exists():
        return pd.read_parquet(year_path)
    legacy_path = vdb_dir / "item_vectors.parquet"
    if legacy_path.exists():
        df = pd.read_parquet(legacy_path)
        if "year" in df.columns:
            return df[df["year"] == int(year)].copy()
        return df
    raise FileNotFoundError(f"Missing {year_path} (and legacy fallback {legacy_path})")


def load_unit_text_by_firm(vdb_dir: Path, year: int, item_id: str, scope: str) -> Dict[str, str]:
    """
    Load unit text from `units.parquet` and return `{firm_id: merged_item_text}` for one item/year.

    Text is assembled in `unit_id` order and used by LLM-based peer comparison.
    """
    year_path = vdb_dir / "units" / f"units_{int(year)}.parquet"
    legacy_path = vdb_dir / "units.parquet"
    units_path = year_path if year_path.exists() else legacy_path
    if not units_path.exists():
        raise FileNotFoundError(f"Missing {year_path} (and legacy fallback {legacy_path})")

    units = pd.read_parquet(units_path)
    if units.empty:
        return {}

    sub = units[(units["year"] == int(year)) & (units["item_id"].astype(str).str.upper() == str(item_id).upper())]
    if "scope" in units.columns:
        sub = sub[sub["scope"].astype(str).str.lower() == str(scope).lower()]
    if sub.empty:
        return {}

    sub = sub.sort_values(["firm_id", "unit_id"])
    grouped = sub.groupby("firm_id", sort=False)["unit_text"].apply(lambda s: "\n\n".join(str(x) for x in s))
    return {str(k): str(v) for k, v in grouped.items()}


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n\n[...TRUNCATED...]\n\n" + text[-keep:]


def _extract_first_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _resolve_llama_api_key(cli_key: Optional[str]) -> str:
    """Resolve llama endpoint API key (optional for local stacks)."""
    key = str(cli_key or os.getenv("LLAMA_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    # Local llama-stack typically ignores API key; keep a non-empty placeholder for client compatibility.
    return key or "local"


def llama3_similarity_score(
    *,
    base_url: str,
    api_key: str,
    model: str,
    focal_firm: str,
    peer_firm: str,
    item_id: str,
    year: int,
    focal_text: str,
    peer_text: str,
    timeout_sec: int,
) -> Tuple[float, str]:
    """
    Ask local llama3 endpoint for item-content similarity score in [0, 1].

    Uses OpenAI-compatible chat completions API exposed by llama-stack.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("openai package is required for llama3 method client calls.") from exc

    prompt = f"""
You are comparing two firms using only SEC 10-K item text.

Task:
- Compare semantic business similarity for item {item_id} in fiscal year {year}.
- Use ONLY supplied text snippets.
- Return JSON object with:
  - "score": number in [0,1]
  - "reason": one short sentence

Firm A (focal): {focal_firm}
Firm B (peer): {peer_firm}

Firm A item text:
{focal_text}

Firm B item text:
{peer_text}
""".strip()

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Return compact JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    obj = _extract_first_json_object(text)
    if not obj:
        return float("nan"), "llama_response_not_json"
    try:
        score = float(obj.get("score"))
    except Exception:
        return float("nan"), str(obj.get("reason") or "missing_score")
    score = max(0.0, min(1.0, score))
    reason = str(obj.get("reason") or "")
    return score, reason[:400]


def run_peerfinder(
    *,
    vdb_dir: Path,
    scope: str,
    focalfirm: str,
    year: int,
    items: List[str],
    top_k: int,
    q_share: float,
    faiss_use_gpu: bool,
    method: str,
    precompute: bool,
    precompute_overwrite: bool,
    llama_base_url: str,
    llama_api_key: Optional[str],
    llama_model: str,
    llama_max_chars: int,
    llama_timeout_sec: int,
) -> pd.DataFrame:
    """
    Run peer search for all requested items and return peer rows.

    Behavior:
    - Cache-first: if precomputed matrix exists, use it.
    - Optional cache build: `precompute=True` builds missing (or all with overwrite).
    - Fallback: on-demand FAISS search if no precomputed matrix is present.
    """
    _warn_if_gpu_available_without_faiss_gpu(faiss_use_gpu)
    active_vdb_dir = _resolve_scope_dir(vdb_dir, scope)
    _ = load_item_vectors(active_vdb_dir, year=year)
    selected_items = sorted({str(item_id).upper() for item_id in items if str(item_id).strip()})

    rows: List[Dict[str, object]] = []

    for item_id in selected_items:
        pooled_path = active_vdb_dir / "vectors" / "pooled" / f"item={item_id}" / f"year={year}.npz"
        residual_path = active_vdb_dir / "vectors" / "residual" / f"item={item_id}" / f"year={year}.npz"
        if not pooled_path.exists():
            continue

        pooled_mat, firm_ids = load_pooled(pooled_path)
        residual_mat = np.zeros_like(pooled_mat)
        residual_mask = np.zeros((pooled_mat.shape[0],), dtype=np.int8)
        if method == "orthogonal":
            if not residual_path.exists():
                continue
            residual_mat, residual_ids, residual_mask = load_residual(residual_path)
            if not np.array_equal(firm_ids, residual_ids):
                raise RuntimeError(f"ID alignment mismatch for item={item_id}, year={year}")

        if method == "llama3":
            if precompute:
                print("[WARN] --precompute is ignored for method=llama3 (LLM scoring runs on demand).")
            api_key = _resolve_llama_api_key(llama_api_key)
            where = np.where(firm_ids == str(focalfirm))[0]
            if len(where) == 0:
                continue

            texts = load_unit_text_by_firm(active_vdb_dir, year=year, item_id=item_id, scope=scope)
            focal_text_raw = texts.get(str(focalfirm), "")
            if not focal_text_raw:
                continue
            focal_text = _truncate_text(focal_text_raw, llama_max_chars)

            scored: List[Tuple[str, float, str]] = []
            for peer_firm in firm_ids:
                peer_id = str(peer_firm)
                if peer_id == str(focalfirm):
                    continue
                peer_text_raw = texts.get(peer_id, "")
                if not peer_text_raw:
                    continue
                peer_text = _truncate_text(peer_text_raw, llama_max_chars)
                try:
                    score, reason = llama3_similarity_score(
                        base_url=llama_base_url,
                        api_key=api_key,
                        model=llama_model,
                        focal_firm=str(focalfirm),
                        peer_firm=peer_id,
                        item_id=item_id,
                        year=int(year),
                        focal_text=focal_text,
                        peer_text=peer_text,
                        timeout_sec=llama_timeout_sec,
                    )
                except Exception as exc:
                    score, reason = float("nan"), f"llama_error: {exc}"
                scored.append((peer_id, score, reason))

            scored = [x for x in scored if not np.isnan(x[1])]
            scored.sort(key=lambda x: x[1], reverse=True)
            for rank, (peer_id, score, reason) in enumerate(scored[:top_k], start=1):
                rows.append(
                    {
                        "focal_firm": str(focalfirm),
                        "year": int(year),
                        "item_id": item_id,
                        "scope": scope,
                        "peer_firm": peer_id,
                        "rank": rank,
                        "sim_common": np.nan,
                        "sim_specific": float(score),
                        "q_share": float(q_share),
                        "k": int(top_k),
                        "method": "llama3",
                        "source": "on_demand",
                        "llama_reason": reason,
                    }
                )
            continue

        if precompute and method in {"orthogonal", "cosine"} and (
            precompute_overwrite or not _precomputed_exists(active_vdb_dir, scope, item_id, year, method)
        ):
            matrix = _compute_similarity_matrix(method, pooled_mat, residual_mat, residual_mask)
            _save_precomputed(active_vdb_dir, scope, item_id, year, method, matrix, firm_ids)

        if method in {"orthogonal", "cosine"} and _precomputed_exists(active_vdb_dir, scope, item_id, year, method):
            matrix, cached_ids = _load_precomputed(active_vdb_dir, scope, item_id, year, method)
            where = np.where(cached_ids == str(focalfirm))[0]
            if len(where) == 0:
                continue
            focal_idx = int(where[0])
            peer_idx = _topk_from_similarity_row(matrix[focal_idx], focal_idx, top_k)
            for rank, j in enumerate(peer_idx, start=1):
                score = float(matrix[focal_idx, int(j)])
                rows.append(
                    {
                        "focal_firm": str(focalfirm),
                        "year": int(year),
                        "item_id": item_id,
                        "scope": scope,
                        "peer_firm": str(cached_ids[int(j)]),
                        "rank": rank,
                        "sim_common": score if method == "cosine" else np.nan,
                        "sim_specific": score if method == "orthogonal" else np.nan,
                        "q_share": float(q_share),
                        "k": int(top_k),
                        "method": method,
                        "source": "precomputed",
                    }
                )
            continue

        where = np.where(firm_ids == str(focalfirm))[0]
        if len(where) == 0:
            continue
        focal_idx = int(where[0])

        if method == "cosine":
            # Direct pairwise cosine on pooled embeddings: cos(E_i, E_j)
            pool_index = build_index(pooled_mat, use_gpu=faiss_use_gpu)
            k_eff = min(top_k + 1, len(firm_ids))
            nn_idx, nn_sims = faiss_topk(pool_index, pooled_mat[focal_idx], k_eff)
            nn_idx = nn_idx[nn_idx >= 0]
            nn_sims = nn_sims[: len(nn_idx)]

            rank = 0
            for local_pos, j in enumerate(nn_idx):
                if int(j) == focal_idx:
                    continue
                rank += 1
                if rank > top_k:
                    break
                rows.append(
                    {
                        "focal_firm": str(focalfirm),
                        "year": int(year),
                        "item_id": item_id,
                        "scope": scope,
                        "peer_firm": str(firm_ids[int(j)]),
                        "rank": rank,
                        "sim_common": float(nn_sims[local_pos]),
                        "sim_specific": np.nan,
                        "q_share": float(q_share),
                        "k": int(top_k),
                        "method": "cosine",
                        "source": "on_demand",
                    }
                )
            continue

        candidates = item_candidates(pooled_mat, focal_idx, q_share=q_share, use_gpu=faiss_use_gpu)
        if len(candidates) == 0:
            continue

        focal_has_residual = bool(residual_mask[focal_idx] == 1)
        if focal_has_residual:
            valid_candidates = candidates[residual_mask[candidates].astype(bool)]
            if len(valid_candidates) > 0:
                cand_mat = residual_mat[valid_candidates]
                idx = build_index(cand_mat, use_gpu=faiss_use_gpu)
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
                            "scope": scope,
                            "peer_firm": str(firm_ids[j]),
                            "rank": rank,
                            "sim_common": float(common_sims[rank - 1]),
                            "sim_specific": float(spec_sims[rank - 1]),
                            "q_share": float(q_share),
                            "k": int(top_k),
                            "method": "orthogonal",
                            "source": "on_demand",
                        }
                    )
                continue

        cand_mat = pooled_mat[candidates]
        idx = build_index(cand_mat, use_gpu=faiss_use_gpu)
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
                    "scope": scope,
                    "peer_firm": str(firm_ids[j]),
                    "rank": rank,
                    "sim_common": float(common_sims[rank - 1]),
                    "sim_specific": np.nan,
                    "q_share": float(q_share),
                    "k": int(top_k),
                    "method": "orthogonal",
                    "source": "on_demand",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No peers produced. Check focal firm/item/year availability in built vectors.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Find peers using common-screen and specific-rank")
    parser.add_argument("--vdb_dir", required=True, help="Output directory generated by vdbbuilder")
    parser.add_argument(
        "--scope",
        default="all",
        choices=["heading", "body", "all"],
        help="Text scope to query. Expects artifacts under vdb_dir/scope=<scope>.",
    )
    parser.add_argument("--focalfirm", required=True, help="Focal firm id/CIK")
    parser.add_argument("--year", type=int, required=True, help="Fiscal year")
    parser.add_argument(
        "--item",
        nargs="+",
        required=True,
        help="One or more item ids to run (e.g., --item 1A 7 7A)",
    )
    parser.add_argument("--k", type=int, default=20, help="Top peers per item")
    parser.add_argument("--q_share", type=float, default=0.20, help="Candidate screening share")
    parser.add_argument(
        "--method",
        default="orthogonal",
        choices=["orthogonal", "cosine", "llama3"],
        help="Similarity method: orthogonal, cosine, or llama3 (LLM text comparison from units.parquet).",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Precompute and save NxN similarity matrix for each requested item/year/method",
    )
    parser.add_argument(
        "--precompute-overwrite",
        action="store_true",
        help="Overwrite existing precomputed similarity matrices",
    )
    parser.add_argument(
        "--faiss-gpu",
        dest="faiss_use_gpu",
        action="store_true",
        default=True,
        help="Use FAISS GPU acceleration when available (default: on)",
    )
    parser.add_argument("--no-faiss-gpu", dest="faiss_use_gpu", action="store_false")
    parser.add_argument(
        "--llama-base-url",
        default="http://localhost:8321/v1",
        help="OpenAI-compatible base URL for llama-stack endpoint.",
    )
    parser.add_argument(
        "--llama-api-key",
        default=None,
        help="Optional API key for llama endpoint. Defaults to LLAMA_API_KEY/OPENAI_API_KEY or 'local'.",
    )
    parser.add_argument(
        "--llama-model",
        default="llama3.3-70b",
        help="Llama model id for --method llama3.",
    )
    parser.add_argument(
        "--llama-max-chars",
        type=int,
        default=12000,
        help="Max chars per firm item text sent to llama3 (head+tail truncation if exceeded).",
    )
    parser.add_argument(
        "--llama-timeout-sec",
        type=int,
        default=120,
        help="HTTP timeout seconds for llama3 requests.",
    )
    parser.add_argument(
        "--out_path",
        default="output/peer_sets_{timestamp}.csv",
        help="Output csv/parquet path. Supports {timestamp} placeholder.",
    )
    args = parser.parse_args()

    out_df = run_peerfinder(
        vdb_dir=Path(args.vdb_dir),
        scope=str(args.scope),
        focalfirm=str(args.focalfirm),
        year=int(args.year),
        items=[str(x) for x in args.item],
        top_k=int(args.k),
        q_share=float(args.q_share),
        faiss_use_gpu=bool(args.faiss_use_gpu),
        method=str(args.method),
        precompute=bool(args.precompute),
        precompute_overwrite=bool(args.precompute_overwrite),
        llama_base_url=str(args.llama_base_url),
        llama_api_key=str(args.llama_api_key) if args.llama_api_key else None,
        llama_model=str(args.llama_model),
        llama_max_chars=int(args.llama_max_chars),
        llama_timeout_sec=int(args.llama_timeout_sec),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(str(args.out_path).replace("{timestamp}", timestamp))
    ensure_dir(out_path)
    if out_path.suffix.lower() == ".parquet":
        out_df.to_parquet(out_path, index=False)
    else:
        low = out_path.name.lower()
        if low.endswith(".csv.gz") or low.endswith(".gz"):
            out_df.to_csv(out_path, index=False, compression="gzip")
        else:
            if out_path.suffix.lower() != ".csv":
                out_path = out_path.with_suffix(".csv")
            out_df.to_csv(out_path, index=False)

    print(f"[DONE] Wrote peer sets: {out_path}")


if __name__ == "__main__":
    main()
