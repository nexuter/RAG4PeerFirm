"""
Build unit table, item vectors, and per-item-per-year vector indices from raw filings.

Algorithm summary:
1) Segment filing text into item sections.
2) Chunk each item into overlapping units.
3) Canonicalize text lightly (spaces + numeric tokens).
4) Embed each unit (OpenAI or local deterministic hash embedder).
5) Pool units into item vectors via distinctiveness weighting.
6) Compute leave-one-out common direction per (item, year).
7) Orthogonally decompose into common loading and residual vectors.
8) Persist parquet tables + npz matrices + FAISS indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

DEFAULT_ITEMS = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15",
]

ITEM_RE = re.compile(r"(?im)^\s*item\s+((?:\d{1,2})(?:[a-d])?)\s*[\.:\-]\s*")
NUM_PATTERNS = [
    (re.compile(r"(?i)\(\s*\$?\s*\d[\d,]*\.?\d*\s*\)"), " NEG_NUM "),
    (re.compile(r"(?i)-\s*\$?\s*\d[\d,]*\.?\d*"), " NEG_NUM "),
    (re.compile(r"(?i)\$?\s*\d[\d,]*\.?\d*\s*%"), " PCT "),
    (re.compile(r"(?i)\$?\s*\d[\d,]*\.?\d*"), " NUM "),
]


class Embedder:
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class LocalHashEmbedder(Embedder):
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


def make_embedder(name: str, model: str, batch_size: int) -> Tuple[Embedder, str]:
    if name == "local":
        emb = LocalHashEmbedder()
        return emb, f"local-hash-{emb.dim}"
    if name == "openai":
        emb = OpenAIEmbedder(model=model, batch_size=batch_size)
        return emb, model
    raise ValueError(f"Unsupported --embedder: {name}")


def normalize_item_id(item: str) -> str:
    return re.sub(r"\s+", "", item.strip().upper())


def canonicalize_text(text: str) -> str:
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


def token_count(text: str) -> int:
    return len(tokenize(text))


def chunk_by_tokens(text: str, chunk_tokens: int, overlap_tokens: int, min_unit_tokens: int) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []

    buffer: List[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer_tokens
        if not buffer:
            return
        candidate = "\n\n".join(buffer).strip()
        if token_count(candidate) >= min_unit_tokens:
            units.append(candidate)
        buffer.clear()
        buffer_tokens = 0

    for p in paras:
        pt = token_count(p)
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


def parse_items(raw_text: str, allowed_items: Sequence[str]) -> Dict[str, str]:
    allowed = {normalize_item_id(x) for x in allowed_items}
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(ITEM_RE.finditer(text))
    if not matches:
        return {}

    spans: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        item = normalize_item_id(match.group(1))
        if item not in allowed:
            continue
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if item not in spans or len(chunk) > len(spans[item]):
            spans[item] = chunk
    return spans


def distinctiveness_weighted_pool(unit_vecs: np.ndarray, eps: float = 1e-6, cap_percentile: float = 95.0) -> Tuple[np.ndarray, Dict[str, float]]:
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
    alpha = float(np.dot(v, direction))
    residual = v - alpha * direction
    norm = float(np.linalg.norm(residual))
    return alpha, residual.astype(np.float32), norm


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def infer_firm_year_from_path(path: Path, filing_dir: Path) -> Tuple[str, int]:
    rel = path.relative_to(filing_dir)
    parts = rel.parts

    if len(parts) >= 4 and parts[1].isdigit() and parts[2].upper() in {"10-K", "10-Q"}:
        return parts[0], int(parts[1])

    m = re.search(r"(\d{10})_(\d{4})_10-K", path.name, flags=re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))

    raise ValueError(f"Unable to infer firm/year from path: {path}")


def list_filing_files(filing_dir: Path, filing_type: str) -> List[Path]:
    expected_type = filing_type.upper()
    candidates: List[Path] = []
    for p in filing_dir.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in {".txt", ".htm", ".html"}:
            continue
        if expected_type not in str(p).upper():
            continue
        candidates.append(p)
    return sorted(candidates)


def save_npz(path: Path, mat: np.ndarray, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mat=mat.astype(np.float32), ids=ids)


def build_faiss_index(mat: np.ndarray) -> "faiss.Index":
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat.astype(np.float32))
    return index


@dataclass
class BuildConfig:
    filing_dir: Path
    out_dir: Path
    filing_type: str
    items: List[str]
    embedder: str
    embed_model: str
    batch_size: int
    chunk_tokens: int
    overlap_tokens: int
    min_unit_tokens: int
    min_units_per_item: int
    residual_norm_floor: float
    cap_weight_percentile: float


def build(cfg: BuildConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    embedder, embed_model_name = make_embedder(cfg.embedder, cfg.embed_model, cfg.batch_size)
    files = list_filing_files(cfg.filing_dir, cfg.filing_type)
    if not files:
        raise RuntimeError(f"No filings found under {cfg.filing_dir}")

    units_rows: List[Dict[str, object]] = []
    item_rows: List[Dict[str, object]] = []

    for idx, path in enumerate(files, start=1):
        try:
            firm_id, year = infer_firm_year_from_path(path, cfg.filing_dir)
        except ValueError:
            continue

        raw = read_text(path)
        item_texts = parse_items(raw, cfg.items)
        if not item_texts:
            continue

        for item_id, item_text in item_texts.items():
            norm = canonicalize_text(item_text)
            units = chunk_by_tokens(norm, cfg.chunk_tokens, cfg.overlap_tokens, cfg.min_unit_tokens)
            units = [u for u in units if token_count(u) >= cfg.min_unit_tokens]
            if len(units) < cfg.min_units_per_item:
                continue

            unit_vecs = embedder.embed_texts(units)
            pooled, pool_stats = distinctiveness_weighted_pool(unit_vecs, cap_percentile=cfg.cap_weight_percentile)

            for uid, unit_text in enumerate(units, start=1):
                units_rows.append(
                    {
                        "firm_id": str(firm_id),
                        "year": int(year),
                        "item_id": item_id,
                        "unit_id": uid,
                        "unit_text": unit_text,
                        "unit_tokens": token_count(unit_text),
                        "embedding_model": embed_model_name,
                        "embedding": json.dumps(unit_vecs[uid - 1].astype(float).tolist()),
                        "source_path": str(path),
                    }
                )

            item_rows.append(
                {
                    "firm_id": str(firm_id),
                    "year": int(year),
                    "item_id": item_id,
                    "num_units": len(units),
                    "item_tokens": sum(token_count(u) for u in units),
                    "embedding_model": embed_model_name,
                    "pooled_embedding": json.dumps(pooled.astype(float).tolist()),
                    "w_max": pool_stats["w_max"],
                    "w_mean": pool_stats["w_mean"],
                    "common_loading": np.nan,
                    "residual_norm": np.nan,
                    "residual_embedding": None,
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

        pooled_path = cfg.out_dir / "vectors" / "pooled" / f"item={item_id}" / f"year={year}.npz"
        save_npz(pooled_path, mat, firm_ids)

        residual_mat = np.zeros_like(mat)
        residual_mask = np.zeros((len(residuals),), dtype=np.int8)
        for i, rv in enumerate(residuals):
            if rv is None:
                continue
            residual_mat[i] = rv
            residual_mask[i] = 1

        residual_path = cfg.out_dir / "vectors" / "residual" / f"item={item_id}" / f"year={year}.npz"
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(residual_path, mat=residual_mat, ids=firm_ids, mask=residual_mask)

        if _HAS_FAISS:
            idx_dir = cfg.out_dir / "indices" / f"item={item_id}" / f"year={year}"
            idx_dir.mkdir(parents=True, exist_ok=True)

            pooled_index = build_faiss_index(mat)
            faiss.write_index(pooled_index, str(idx_dir / "pooled.faiss"))
            (idx_dir / "pooled_ids.json").write_text(json.dumps(firm_ids.tolist()), encoding="utf-8")

            valid = residual_mask.astype(bool)
            if valid.sum() > 0:
                residual_index = build_faiss_index(residual_mat[valid])
                faiss.write_index(residual_index, str(idx_dir / "residual.faiss"))
                (idx_dir / "residual_ids.json").write_text(json.dumps(firm_ids[valid].tolist()), encoding="utf-8")

    units_out = cfg.out_dir / "units.parquet"
    items_out = cfg.out_dir / "item_vectors.parquet"
    units_df.to_parquet(units_out, index=False)
    items_df.drop(columns=["_row_idx"]).to_parquet(items_out, index=False)

    print(f"[DONE] Wrote: {units_out}")
    print(f"[DONE] Wrote: {items_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build vector DB for item-level peer matching")
    ap.add_argument("--filing_dir", required=True, help="Root directory containing downloaded raw filings")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--filing", default="10-K", choices=["10-K", "10-Q"], dest="filing_type")
    ap.add_argument("--items", default=",".join(DEFAULT_ITEMS), help="Comma-separated item ids")

    ap.add_argument("--embedder", default="openai", choices=["openai", "local"], help="Embedding backend")
    ap.add_argument("--embed_model", default="text-embedding-3-large", help="OpenAI embedding model")
    ap.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")

    ap.add_argument("--chunk_tokens", type=int, default=280)
    ap.add_argument("--overlap_tokens", type=int, default=60)
    ap.add_argument("--min_unit_tokens", type=int, default=80)
    ap.add_argument("--min_units_per_item", type=int, default=3)

    ap.add_argument("--residual_norm_floor", type=float, default=0.10)
    ap.add_argument("--cap_weight_percentile", type=float, default=95.0)

    args = ap.parse_args()

    cfg = BuildConfig(
        filing_dir=Path(args.filing_dir),
        out_dir=Path(args.out_dir),
        filing_type=args.filing_type,
        items=[normalize_item_id(i) for i in args.items.split(",") if i.strip()],
        embedder=args.embedder,
        embed_model=args.embed_model,
        batch_size=int(args.batch_size),
        chunk_tokens=int(args.chunk_tokens),
        overlap_tokens=int(args.overlap_tokens),
        min_unit_tokens=int(args.min_unit_tokens),
        min_units_per_item=int(args.min_units_per_item),
        residual_norm_floor=float(args.residual_norm_floor),
        cap_weight_percentile=float(args.cap_weight_percentile),
    )

    build(cfg)


if __name__ == "__main__":
    main()
