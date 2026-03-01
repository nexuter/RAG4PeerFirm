"""
vdbbuilder.py

Builds vectorized item-level data artifacts from extracted SEC filing JSON files.

Input:
- Filing files: `*_item.json` under `<filing_dir>/<firm_id>/<year>/<filing_type>/`.
- Each file is expected to contain:
  - `toc_items`: item ids available in the filing.
  - `items[item_id].text_content` (or fallback `html_content`).

Core pipeline design:
1. Read filing item text from JSON.
2. Canonicalize text and split into token-based overlapping units.
3. Embed units with selected backend:
   - `local`: deterministic hash embedding (offline, reproducible).
   - `openai`: OpenAI embedding API.
4. Pool unit vectors into one item vector using distinctiveness weighting.
5. For each (item, year), compute leave-one-out common direction and residual vector.
6. Persist tables and matrices; optionally write FAISS indices.

Outputs under `--out_dir`:
- `units.parquet`: one row per unit with metadata and unit embedding.
- `item_vectors.parquet`: one row per firm-year-item with pooled and residual stats.
- `vectors/pooled/item=<ITEM>/year=<YEAR>.npz`: normalized pooled vectors + ids.
- `vectors/residual/item=<ITEM>/year=<YEAR>.npz`: normalized residual vectors + ids + mask.
- `indices/...` (optional): FAISS inner-product indices for pooled/residual vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

DEFAULT_ITEMS = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15",
]
DEFAULT_WEIGHT_EPS = 1e-6

NUM_PATTERNS = [
    (re.compile(r"(?i)\(\s*\$?\s*\d[\d,]*\.?\d*\s*\)"), " NEG_NUM "),
    (re.compile(r"(?i)-\s*\$?\s*\d[\d,]*\.?\d*"), " NEG_NUM "),
    (re.compile(r"(?i)\$?\s*\d[\d,]*\.?\d*\s*%"), " PCT "),
    (re.compile(r"(?i)\$?\s*\d[\d,]*\.?\d*"), " NUM "),
]


class Embedder:
    """Embedding interface for unit text batches."""

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class LocalHashEmbedder(Embedder):
    """Deterministic local embedder for offline end-to-end execution."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts], axis=0)


class OpenAIEmbedder(Embedder):
    """OpenAI API embedder with batched requests."""

    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package not installed. pip install openai") from exc
        self.client = OpenAI()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            vectors.append(arr)
        return np.concatenate(vectors, axis=0)


def make_embedder(name: str, model: Optional[str], batch_size: int) -> Tuple[Embedder, str]:
    """Create embedder instance and return `(embedder, model_name_for_metadata)`."""
    if name == "local":
        emb = LocalHashEmbedder()
        return emb, f"local-hash-{emb.dim}"
    if name == "openai":
        if not model:
            raise ValueError("--embed_model is required when --embedder openai is used")
        emb = OpenAIEmbedder(model=model, batch_size=batch_size)
        return emb, model
    raise ValueError(f"Unsupported --embedder: {name}")


def normalize_item_id(item: str) -> str:
    return re.sub(r"\s+", "", item.strip().upper())


class TokenCounter:
    """
    Token counter used for chunking and unit filtering.

    Uses `tiktoken` when available; falls back to whitespace tokenization.
    """

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model_name = model_name
        self._enc = None
        if _HAS_TIKTOKEN:
            try:
                self._enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                self._enc = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return len(tokenize(text))


def canonicalize_text(text: str) -> str:
    """
    Light text normalization for robust chunking and embeddings.

    Keeps paragraph breaks, collapses whitespace, and canonicalizes numeric patterns.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " | ", text)
    for pattern, replacement in NUM_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+\|\s+\|\s+", " | ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def chunk_by_tokens(
    text: str,
    token_counter: TokenCounter,
    chunk_tokens: int,
    overlap_tokens: int,
    min_unit_tokens: int,
) -> List[str]:
    """
    Split item text into overlapping token-based units.

    Strategy:
    - Respect paragraph boundaries where possible.
    - Hard-split long paragraphs by token windows.
    - Apply minimum token filter at unit level.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []

    buffer: List[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer_tokens
        if not buffer:
            return
        candidate = "\n\n".join(buffer).strip()
        if token_counter.count(candidate) >= min_unit_tokens:
            units.append(candidate)
        buffer.clear()
        buffer_tokens = 0

    for p in paras:
        pt = token_counter.count(p)
        if pt >= chunk_tokens:
            words = tokenize(p)
            step = max(1, chunk_tokens - overlap_tokens)
            for start in range(0, len(words), step):
                w = words[start : start + chunk_tokens]
                if len(w) >= min_unit_tokens:
                    units.append(" ".join(w))
            continue

        if buffer_tokens + pt <= chunk_tokens:
            buffer.append(p)
            buffer_tokens += pt
        else:
            flush()
            buffer.append(p)
            buffer_tokens = pt

    flush()
    return units


def distinctiveness_weighted_pool(
    unit_vecs: np.ndarray,
    eps: float = DEFAULT_WEIGHT_EPS,
    cap_percentile: float = 95.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Pool unit vectors into one item vector using distance-from-centroid weights.

    Returns:
    - L2-normalized pooled vector.
    - Weight statistics (`w_max`, `w_mean`) for diagnostics.
    """
    if unit_vecs.shape[0] == 1:
        vec = unit_vecs[0]
        return vec, {"w_max": 1.0, "w_mean": 1.0}

    centroid = unit_vecs.mean(axis=0)
    weights = np.linalg.norm(unit_vecs - centroid[None, :], axis=1) + eps
    cap = np.percentile(weights, cap_percentile)
    weights = np.minimum(weights, cap)

    pooled = (weights[:, None] * unit_vecs).sum(axis=0) / (weights.sum() + 1e-12)
    pooled = pooled.astype(np.float32)
    pooled /= np.linalg.norm(pooled) + 1e-12

    return pooled, {"w_max": float(weights.max()), "w_mean": float(weights.mean())}


def orthogonal_decompose(v: np.ndarray, direction: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """Return `(alpha, residual, residual_norm)` from projection of `v` onto `direction`."""
    alpha = float(np.dot(v, direction))
    residual = v - alpha * direction
    norm = float(np.linalg.norm(residual))
    return alpha, residual.astype(np.float32), norm


def infer_firm_year_from_path(path: Path, filing_dir: Path) -> Tuple[str, int]:
    """Infer `(firm_id, year)` from canonical sec_filings layout or filename fallback."""
    rel = path.relative_to(filing_dir)
    parts = rel.parts

    if len(parts) >= 4 and parts[1].isdigit() and parts[2].upper() in {"10-K", "10-Q"}:
        return parts[0], int(parts[1])

    m = re.search(r"(\d{10})_(\d{4})_10-K", path.name, flags=re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))

    raise ValueError(f"Unable to infer firm/year from path: {path}")


def list_filing_files(filing_dir: Path, filing_type: str, scope: str) -> List[Path]:
    """List source JSON files by scope and filing type."""
    expected_type = filing_type.upper()
    suffix = "_item.json" if scope == "all" else "_str.json"
    candidates: List[Path] = []
    for p in filing_dir.rglob(f"*{suffix}"):
        if not p.is_file():
            continue
        if expected_type not in str(p).upper():
            continue
        candidates.append(p)
    return sorted(candidates)


def extract_text(s: str) -> str:
    soup = BeautifulSoup(s, "lxml")
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def load_items_from_json(path: Path, allowed_items: Sequence[str]) -> Dict[str, str]:
    """
    Extract `{item_id: text_content}` for requested items from one filing JSON file.

    Item selection is driven by `toc_items` keys; text is read from `items`.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    toc_items = payload.get("toc_items")
    items = payload.get("items")
    if not isinstance(toc_items, dict) or not isinstance(items, dict):
        return {}
    allowed = {normalize_item_id(x) for x in allowed_items}
    out: Dict[str, str] = {}
    # Drive selection from toc_items, then read text_content from items[item_id].
    for key in toc_items.keys():
        item_id = normalize_item_id(str(key))
        if item_id not in allowed:
            continue
        item_payload = items.get(str(key)) or items.get(item_id)
        if not isinstance(item_payload, dict):
            continue
        text = str(item_payload.get("text_content") or "").strip()
        if not text:
            html = str(item_payload.get("html_content") or "")
            if html:
                text = extract_text(html)
        if text:
            out[item_id] = text
    return out


def _collect_node_values(node: object, field: str, out: List[str]) -> None:
    """Recursively collect `heading` or `body` values from nested structures."""
    if isinstance(node, dict):
        val = node.get(field)
        if isinstance(val, str):
            s = val.strip()
            if s:
                out.append(s)
        children = node.get("children")
        if isinstance(children, list):
            for ch in children:
                _collect_node_values(ch, field, out)
        return
    if isinstance(node, list):
        for x in node:
            _collect_node_values(x, field, out)


def load_items_from_struct_json(path: Path, allowed_items: Sequence[str], scope: str) -> Dict[str, str]:
    """
    Extract item text from `*_str.json` structures.

    - scope=heading: concatenate all `heading` values in each item subtree.
    - scope=body: concatenate all `body` values in each item subtree.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    structures = payload.get("structures")
    if not isinstance(structures, dict):
        return {}

    allowed = {normalize_item_id(x) for x in allowed_items}
    by_norm: Dict[str, object] = {normalize_item_id(str(k)): v for k, v in structures.items()}
    field = "heading" if scope == "heading" else "body"

    out: Dict[str, str] = {}
    for item_id in allowed:
        root = by_norm.get(item_id)
        if root is None:
            continue
        parts: List[str] = []
        _collect_node_values(root, field, parts)
        if parts:
            out[item_id] = "\n\n".join(parts)
    return out


def save_npz(path: Path, mat: np.ndarray, ids: np.ndarray) -> None:
    """Save matrix and aligned ids as compressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mat=mat.astype(np.float32), ids=ids)


def _faiss_gpu_available() -> bool:
    if not _HAS_FAISS:
        return False
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


def _faiss_to_cpu(index: "faiss.Index") -> "faiss.Index":
    if hasattr(faiss, "index_gpu_to_cpu"):
        try:
            return faiss.index_gpu_to_cpu(index)
        except Exception:
            return index
    return index


def build_faiss_index(mat: np.ndarray, use_gpu: bool = True) -> "faiss.Index":
    cpu_index = faiss.IndexFlatIP(mat.shape[1])
    vecs = mat.astype(np.float32)
    if use_gpu and _faiss_gpu_available():
        try:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
            gpu_index.add(vecs)
            return gpu_index
        except Exception:
            pass
    cpu_index.add(vecs)
    return cpu_index


@dataclass
class BuildConfig:
    """Configuration for the full vector-db build pipeline."""

    filing_dir: Path
    out_dir: Path
    filing_type: str
    scope: str
    items: List[str]
    embedder: str
    embed_model: Optional[str]
    batch_size: int
    tokenizer_model: str
    chunk_tokens: int
    overlap_tokens: int
    min_unit_tokens: int
    min_units_per_item: int
    residual_norm_floor: float
    cap_weight_percentile: float
    build_faiss: bool
    faiss_use_gpu: bool


def build(cfg: BuildConfig) -> None:
    """
    Execute full build pipeline.

    Input:
    - `cfg` controls paths, embedding backend, chunking, decomposition, and FAISS options.

    Main outputs:
    - `units.parquet`
    - `item_vectors.parquet`
    - pooled/residual `.npz` matrices per item-year
    - optional FAISS index files per item-year
    """
    scoped_out_dir = cfg.out_dir / f"scope={cfg.scope}"
    scoped_out_dir.mkdir(parents=True, exist_ok=True)

    embedder, embed_model_name = make_embedder(cfg.embedder, cfg.embed_model, cfg.batch_size)
    token_counter = TokenCounter(model_name=cfg.tokenizer_model)
    _warn_if_gpu_available_without_faiss_gpu(cfg.faiss_use_gpu)
    files = list_filing_files(cfg.filing_dir, cfg.filing_type, cfg.scope)

    if not files:
        raise RuntimeError(f"No source JSON filings found for scope={cfg.scope} under {cfg.filing_dir}")

    units_rows: List[Dict[str, object]] = []
    item_rows: List[Dict[str, object]] = []

    for idx, path in enumerate(files, start=1):
        try:
            firm_id, year = infer_firm_year_from_path(path, cfg.filing_dir)
        except ValueError:
            continue

        if cfg.scope == "all":
            item_texts = load_items_from_json(path, cfg.items)
        else:
            item_texts = load_items_from_struct_json(path, cfg.items, cfg.scope)
        if not item_texts:
            continue

        for item_id, item_text in item_texts.items():
            norm = canonicalize_text(item_text)
            units = chunk_by_tokens(
                norm,
                token_counter,
                cfg.chunk_tokens,
                cfg.overlap_tokens,
                cfg.min_unit_tokens,
            )
            units = [u for u in units if token_counter.count(u) >= cfg.min_unit_tokens]
            if len(units) < cfg.min_units_per_item:
                continue

            unit_vecs = embedder.embed_texts(units)
            pooled, pool_stats = distinctiveness_weighted_pool(
                unit_vecs,
                eps=DEFAULT_WEIGHT_EPS,
                cap_percentile=cfg.cap_weight_percentile,
            )

            for uid, unit_text in enumerate(units, start=1):
                units_rows.append(
                    {
                        "firm_id": str(firm_id),
                        "year": int(year),
                        "item_id": item_id,
                        "unit_id": uid,
                        "unit_text": unit_text,
                        "unit_tokens": token_counter.count(unit_text),
                        "embedding_model": embed_model_name,
                        "embedding": json.dumps(unit_vecs[uid - 1].astype(float).tolist()),
                        "source_path": str(path),
                        "scope": cfg.scope,
                    }
                )

            item_rows.append(
                {
                    "firm_id": str(firm_id),
                    "year": int(year),
                    "item_id": item_id,
                    "num_units": len(units),
                    "item_tokens": sum(token_counter.count(u) for u in units),
                    "embedding_model": embed_model_name,
                    "pooled_embedding": json.dumps(pooled.astype(float).tolist()),
                    "w_max": pool_stats["w_max"],
                    "w_mean": pool_stats["w_mean"],
                    "common_loading": np.nan,
                    "residual_norm": np.nan,
                    "residual_embedding": None,
                    "scope": cfg.scope,
                }
            )

        if idx % 100 == 0:
            print(f"[INFO] Processed {idx}/{len(files)} files")

    units_df = pd.DataFrame(units_rows)
    items_df = pd.DataFrame(item_rows)
    if units_df.empty or items_df.empty:
        raise RuntimeError("No units/items built. Check filing directory and item parser.")

    items_df["_row_idx"] = np.arange(len(items_df))

    for item_id, year in sorted(set((r["item_id"], int(r["year"])) for _, r in items_df.iterrows())):
        sub = items_df[(items_df["item_id"] == item_id) & (items_df["year"] == year)]
        idxs = sub["_row_idx"].to_numpy()
        if len(idxs) == 0:
            continue

        mat = np.stack(
            [
                np.array(json.loads(items_df.loc[items_df["_row_idx"] == rid, "pooled_embedding"].values[0]), dtype=np.float32)
                for rid in idxs
            ],
            axis=0,
        )
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12

        sum_all = mat.sum(axis=0)
        alphas: List[float] = []
        residuals: List[Optional[np.ndarray]] = []
        residual_norms: List[float] = []

        for i in range(len(mat)):
            v = mat[i]
            if len(mat) == 1:
                direction = v
            else:
                loo = (sum_all - v) / (len(mat) - 1)
                direction = loo / (np.linalg.norm(loo) + 1e-12)

            alpha, residual, rnorm = orthogonal_decompose(v, direction)
            alphas.append(alpha)
            residual_norms.append(rnorm)
            residuals.append(None if rnorm < cfg.residual_norm_floor else residual / (rnorm + 1e-12))

        sub_idx = sub.index.to_numpy()
        items_df.loc[sub_idx, "common_loading"] = np.array(alphas, dtype=float)
        items_df.loc[sub_idx, "residual_norm"] = np.array(residual_norms, dtype=float)
        items_df.loc[sub_idx, "residual_embedding"] = [
            None if residuals[i] is None else json.dumps(residuals[i].astype(float).tolist())
            for i in range(len(residuals))
        ]

        firm_ids = sub["firm_id"].astype(str).to_numpy()

        pooled_path = scoped_out_dir / "vectors" / "pooled" / f"item={item_id}" / f"year={year}.npz"
        save_npz(pooled_path, mat, firm_ids)

        residual_mat = np.zeros_like(mat)
        residual_mask = np.zeros((len(residuals),), dtype=np.int8)
        for i, rv in enumerate(residuals):
            if rv is None:
                continue
            residual_mat[i] = rv
            residual_mask[i] = 1

        residual_path = scoped_out_dir / "vectors" / "residual" / f"item={item_id}" / f"year={year}.npz"
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(residual_path, mat=residual_mat, ids=firm_ids, mask=residual_mask)

        if cfg.build_faiss:
            if not _HAS_FAISS:
                raise RuntimeError("FAISS index build requested but faiss is not installed.")
            idx_dir = scoped_out_dir / "indices" / f"item={item_id}" / f"year={year}"
            idx_dir.mkdir(parents=True, exist_ok=True)

            pooled_index = build_faiss_index(mat, use_gpu=cfg.faiss_use_gpu)
            faiss.write_index(_faiss_to_cpu(pooled_index), str(idx_dir / "pooled.faiss"))
            (idx_dir / "pooled_ids.json").write_text(json.dumps(firm_ids.tolist()), encoding="utf-8")

            valid = residual_mask.astype(bool)
            if valid.sum() > 0:
                residual_index = build_faiss_index(residual_mat[valid], use_gpu=cfg.faiss_use_gpu)
                faiss.write_index(_faiss_to_cpu(residual_index), str(idx_dir / "residual.faiss"))
                (idx_dir / "residual_ids.json").write_text(json.dumps(firm_ids[valid].tolist()), encoding="utf-8")

    units_out = scoped_out_dir / "units.parquet"
    items_out = scoped_out_dir / "item_vectors.parquet"
    units_df.to_parquet(units_out, index=False)
    items_df.drop(columns=["_row_idx"]).to_parquet(items_out, index=False)

    print(f"[DONE] Wrote: {units_out}")
    print(f"[DONE] Wrote: {items_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build vector DB for item-level peer matching")
    ap.add_argument(
        "--filing_dir",
        default="sec_filings",
        help="Root directory containing extracted filing JSON files (default: sec_filings)",
    )
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--filing", default="10-K", choices=["10-K", "10-Q"], dest="filing_type")
    ap.add_argument(
        "--scope",
        default="all",
        choices=["heading", "body", "all"],
        help="Text scope: all uses *_item.json text_content; heading/body use *_str.json structures.",
    )
    ap.add_argument("--items", default=",".join(DEFAULT_ITEMS), help="Comma-separated item ids")

    ap.add_argument("--embedder", default="local", choices=["openai", "local"], help="Embedding backend")
    ap.add_argument(
        "--embed_model",
        default=None,
        help="OpenAI embedding model (required only when --embedder openai)",
    )
    ap.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    ap.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer model for token counting (uses tiktoken when available)",
    )

    ap.add_argument("--chunk_tokens", type=int, default=280)
    ap.add_argument("--overlap_tokens", type=int, default=60)
    ap.add_argument("--min_unit_tokens", type=int, default=80)
    ap.add_argument("--min_units_per_item", type=int, default=3)

    ap.add_argument("--residual_norm_floor", type=float, default=0.10)
    ap.add_argument("--cap_weight_percentile", type=float, default=95.0)
    ap.add_argument("--build-faiss", dest="build_faiss", action="store_true", default=True)
    ap.add_argument("--no-build-faiss", dest="build_faiss", action="store_false")
    ap.add_argument(
        "--faiss-gpu",
        dest="faiss_use_gpu",
        action="store_true",
        default=True,
        help="Use FAISS GPU acceleration when available (default: on)",
    )
    ap.add_argument("--no-faiss-gpu", dest="faiss_use_gpu", action="store_false")

    args = ap.parse_args()

    cfg = BuildConfig(
        filing_dir=Path(args.filing_dir),
        out_dir=Path(args.out_dir),
        filing_type=args.filing_type,
        scope=str(args.scope),
        items=[normalize_item_id(i) for i in args.items.split(",") if i.strip()],
        embedder=args.embedder,
        embed_model=str(args.embed_model) if args.embed_model else None,
        batch_size=int(args.batch_size),
        tokenizer_model=str(args.tokenizer_model),
        chunk_tokens=int(args.chunk_tokens),
        overlap_tokens=int(args.overlap_tokens),
        min_unit_tokens=int(args.min_unit_tokens),
        min_units_per_item=int(args.min_units_per_item),
        residual_norm_floor=float(args.residual_norm_floor),
        cap_weight_percentile=float(args.cap_weight_percentile),
        build_faiss=bool(args.build_faiss),
        faiss_use_gpu=bool(args.faiss_use_gpu),
    )

    build(cfg)


if __name__ == "__main__":
    main()
