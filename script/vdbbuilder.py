"""
vdbbuilder.py

Build local FAISS-backed vector artifacts for item-level peer matching.

Main design:
1. Load extracted item text from filing JSON files.
2. For `all` and `heading` scopes, chunk long item text, embed chunks locally,
   and pool them into one vector per `(firm, year, item)`.
3. For `summary` scope, read sibling `_summ.json` artifacts and embed each
   summary directly into one vector per `(firm, year, item)`.
4. Persist per-year tables and per-item/year pooled matrices plus FAISS indices.

Current VDB granularity:
- Exactly one pooled vector per `(firm, year, item)` in every scope.

Outputs under `--out_dir/scope=<scope>/`:
- `item_vectors/item_vectors_<YEAR>.parquet`
- `units/units_<YEAR>.parquet` for chunked scopes (`all`, `heading`)
- `vectors/pooled/item=<ITEM>/year=<YEAR>.npz`
- `indices/item=<ITEM>/year=<YEAR>/pooled.faiss`
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

try:
    import faiss  # type: ignore
except Exception as exc:
    raise RuntimeError("faiss is required for vdbbuilder. Install faiss-cpu or faiss-gpu.") from exc

try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


DEFAULT_ITEMS = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16",
]
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_WEIGHT_EPS = 1e-6

# Tuned from observed item length statistics supplied for 10-K items.
ITEM_DEFAULT_CHUNK_TOKENS: Dict[str, int] = {
    "1": 400,
    "1A": 480,
    "1B": 160,
    "1C": 160,
    "2": 240,
    "3": 240,
    "4": 160,
    "5": 280,
    "6": 240,
    "7": 400,
    "7A": 240,
    "8": 480,
    "9": 160,
    "9A": 240,
    "9B": 160,
    "9C": 160,
    "10": 280,
    "11": 240,
    "12": 240,
    "13": 240,
    "14": 200,
    "15": 320,
    "16": 240,
}


class SentenceTransformerEmbedder:
    """Local embedding wrapper around sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for local embedding. "
                "Install it with `pip install sentence-transformers`."
            ) from exc
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        arr = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)


class TokenCounter:
    """Token counter for chunk sizing with tiktoken fallback."""

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self._enc = None
        if _HAS_TIKTOKEN:
            try:
                self._enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                self._enc = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return len(re.findall(r"\S+", text))


def normalize_item_id(item: str) -> str:
    return re.sub(r"\s+", "", item.strip().upper())


def resolve_chunk_tokens(item_id: str, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    return ITEM_DEFAULT_CHUNK_TOKENS.get(normalize_item_id(item_id), 280)


def resolve_overlap_tokens(item_id: str, chunk_tokens: int, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    item_default = ITEM_DEFAULT_CHUNK_TOKENS.get(normalize_item_id(item_id), chunk_tokens)
    scaled = int(round(item_default * 0.2))
    return max(40, min(120, scaled))


def canonicalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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

    for paragraph in paras:
        para_tokens = token_counter.count(paragraph)
        if para_tokens >= chunk_tokens:
            words = tokenize(paragraph)
            step = max(1, chunk_tokens - overlap_tokens)
            for start in range(0, len(words), step):
                window = words[start : start + chunk_tokens]
                if len(window) >= min_unit_tokens:
                    units.append(" ".join(window))
            continue

        if buffer_tokens + para_tokens <= chunk_tokens:
            buffer.append(paragraph)
            buffer_tokens += para_tokens
        else:
            flush()
            buffer.append(paragraph)
            buffer_tokens = para_tokens

    flush()
    return units


def distinctiveness_weighted_pool(
    unit_vecs: np.ndarray,
    eps: float = DEFAULT_WEIGHT_EPS,
    cap_percentile: float = 95.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if unit_vecs.shape[0] == 1:
        vec = unit_vecs[0].astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec, {"w_max": 1.0, "w_mean": 1.0}

    centroid = unit_vecs.mean(axis=0)
    weights = np.linalg.norm(unit_vecs - centroid[None, :], axis=1) + eps
    cap = np.percentile(weights, cap_percentile)
    weights = np.minimum(weights, cap)
    pooled = (weights[:, None] * unit_vecs).sum(axis=0) / (weights.sum() + 1e-12)
    pooled = pooled.astype(np.float32)
    pooled /= np.linalg.norm(pooled) + 1e-12
    return pooled, {"w_max": float(weights.max()), "w_mean": float(weights.mean())}


def extract_text(s: str) -> str:
    soup = BeautifulSoup(s, "lxml")
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def infer_firm_year_from_path(path: Path, filing_dir: Path, filing_type: str) -> Tuple[str, int]:
    rel = path.relative_to(filing_dir)
    parts = rel.parts
    if len(parts) >= 4 and parts[1].isdigit() and parts[2].upper() == filing_type.upper():
        return parts[0], int(parts[1])

    filing_pat = re.escape(filing_type.upper())
    stem = path.stem.upper()
    match = re.search(rf"(\d{{10}})_(\d{{4}})_{filing_pat}", stem)
    if match:
        return match.group(1), int(match.group(2))
    raise ValueError(f"Unable to infer firm/year from path: {path}")


def list_filing_files(filing_dir: Path, filing_type: str, scope: str) -> List[Path]:
    suffix = {
        "all": "_item.json",
        "heading": "_str.json",
        "summary": "_item_summ.json",
    }[scope]
    expected = filing_type.upper()
    files: List[Path] = []
    for path in filing_dir.rglob(f"*{suffix}"):
        if path.is_file() and expected in str(path).upper():
            files.append(path)
    return sorted(files)


def load_items_from_json(path: Path, allowed_items: Sequence[str]) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    toc_items = payload.get("toc_items")
    items = payload.get("items")
    if not isinstance(toc_items, dict) or not isinstance(items, dict):
        return {}
    allowed = {normalize_item_id(x) for x in allowed_items}
    out: Dict[str, str] = {}
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


def load_items_from_struct_json(path: Path, allowed_items: Sequence[str]) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    structures = payload.get("structures")
    if not isinstance(structures, dict):
        return {}
    allowed = {normalize_item_id(x) for x in allowed_items}
    by_norm = {normalize_item_id(str(k)): v for k, v in structures.items()}
    out: Dict[str, str] = {}
    for item_id in allowed:
        root = by_norm.get(item_id)
        if root is None:
            continue
        parts: List[str] = []
        _collect_node_values(root, "heading", parts)
        if parts:
            out[item_id] = "\n\n".join(parts)
    return out


def load_items_from_summary_json(path: Path, allowed_items: Sequence[str]) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict):
        return {}
    allowed = {normalize_item_id(x) for x in allowed_items}
    out: Dict[str, str] = {}
    for key, value in items.items():
        item_id = normalize_item_id(str(key))
        if item_id not in allowed or not isinstance(value, dict):
            continue
        summary = str(value.get("summary") or "").strip()
        if summary:
            out[item_id] = summary
    return out


def save_npz(path: Path, mat: np.ndarray, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mat=mat.astype(np.float32), ids=np.asarray(ids, dtype=np.str_))


def _faiss_gpu_available() -> bool:
    required = ("get_num_gpus", "index_cpu_to_all_gpus")
    if not all(hasattr(faiss, name) for name in required):
        return False
    try:
        return int(faiss.get_num_gpus()) > 0
    except Exception:
        return False


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
    filing_dir: Path
    out_dir: Path
    filing_type: str
    years: Optional[List[int]]
    scope: str
    items: List[str]
    embed_model: str
    tokenizer_model: str
    chunk_tokens: Optional[int]
    overlap_tokens: Optional[int]
    min_unit_tokens: int
    min_units_per_item: int
    faiss_use_gpu: bool
    overwrite: bool


def build(cfg: BuildConfig) -> None:
    if not cfg.filing_dir.exists():
        raise FileNotFoundError(f"Filing directory does not exist: {cfg.filing_dir}")
    if not cfg.filing_dir.is_dir():
        raise NotADirectoryError(f"Filing path is not a directory: {cfg.filing_dir}")

    scoped_out_dir = cfg.out_dir / f"scope={cfg.scope}"
    if cfg.overwrite and scoped_out_dir.exists():
        shutil.rmtree(scoped_out_dir)
    scoped_out_dir.mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformerEmbedder(cfg.embed_model)
    token_counter = TokenCounter(model_name=cfg.tokenizer_model)
    files = list_filing_files(cfg.filing_dir, cfg.filing_type, cfg.scope)
    if not files:
        raise RuntimeError(f"No source JSON filings found for scope={cfg.scope} under {cfg.filing_dir}")

    if cfg.years:
        selected = set(int(y) for y in cfg.years)
        filtered: List[Path] = []
        for path in files:
            try:
                _, year = infer_firm_year_from_path(path, cfg.filing_dir, cfg.filing_type)
            except ValueError:
                continue
            if year in selected:
                filtered.append(path)
        files = filtered
        if not files:
            raise RuntimeError(f"No source JSON filings found for selected years: {sorted(selected)}")

    units_rows: List[Dict[str, object]] = []
    item_rows: List[Dict[str, object]] = []

    for idx, path in enumerate(files, start=1):
        try:
            firm_id, year = infer_firm_year_from_path(path, cfg.filing_dir, cfg.filing_type)
        except ValueError:
            continue

        if cfg.scope == "all":
            item_texts = load_items_from_json(path, cfg.items)
        elif cfg.scope == "heading":
            item_texts = load_items_from_struct_json(path, cfg.items)
        else:
            item_texts = load_items_from_summary_json(path, cfg.items)

        if not item_texts:
            continue

        for item_id, raw_text in item_texts.items():
            text = canonicalize_text(raw_text)
            if not text:
                continue

            if cfg.scope == "summary":
                pooled = embedder.embed_texts([text])[0]
                pooled = pooled.astype(np.float32)
                pooled /= np.linalg.norm(pooled) + 1e-12
                num_units = 1
                item_tokens = token_counter.count(text)
                chunk_tokens = None
                overlap_tokens = None
                w_max = 1.0
                w_mean = 1.0
            else:
                chunk_tokens = resolve_chunk_tokens(item_id, cfg.chunk_tokens)
                overlap_tokens = resolve_overlap_tokens(item_id, chunk_tokens, cfg.overlap_tokens)
                units = chunk_by_tokens(
                    text,
                    token_counter,
                    chunk_tokens,
                    overlap_tokens,
                    cfg.min_unit_tokens,
                )
                units = [unit for unit in units if token_counter.count(unit) >= cfg.min_unit_tokens]
                if len(units) < cfg.min_units_per_item:
                    continue
                unit_vecs = embedder.embed_texts(units)
                pooled, stats = distinctiveness_weighted_pool(unit_vecs)
                num_units = len(units)
                item_tokens = sum(token_counter.count(unit) for unit in units)
                w_max = stats["w_max"]
                w_mean = stats["w_mean"]
                for unit_id, unit_text in enumerate(units, start=1):
                    units_rows.append(
                        {
                            "firm_id": str(firm_id),
                            "year": int(year),
                            "item_id": item_id,
                            "unit_id": unit_id,
                            "unit_text": unit_text,
                            "unit_tokens": token_counter.count(unit_text),
                            "embedding_model": cfg.embed_model,
                            "source_path": str(path),
                            "scope": cfg.scope,
                            "chunk_tokens": chunk_tokens,
                            "overlap_tokens": overlap_tokens,
                        }
                    )

            item_rows.append(
                {
                    "firm_id": str(firm_id),
                    "year": int(year),
                    "item_id": item_id,
                    "embedding_model": cfg.embed_model,
                    "num_units": num_units,
                    "item_tokens": item_tokens,
                    "pooled_embedding": json.dumps(pooled.astype(float).tolist()),
                    "chunk_tokens": chunk_tokens,
                    "overlap_tokens": overlap_tokens,
                    "w_max": w_max,
                    "w_mean": w_mean,
                    "source_path": str(path),
                    "scope": cfg.scope,
                }
            )

        if idx % 100 == 0:
            print(f"[INFO] Processed {idx}/{len(files)} files")

    items_df = pd.DataFrame(item_rows)
    if items_df.empty:
        raise RuntimeError("No item vectors built. Check source files and selected items.")

    units_df = pd.DataFrame(units_rows)
    items_df["_row_idx"] = np.arange(len(items_df))

    for item_id, year in sorted(set((row["item_id"], int(row["year"])) for _, row in items_df.iterrows())):
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
        firm_ids = sub["firm_id"].astype(str).to_numpy()

        pooled_path = scoped_out_dir / "vectors" / "pooled" / f"item={item_id}" / f"year={year}.npz"
        save_npz(pooled_path, mat, firm_ids)

        index_dir = scoped_out_dir / "indices" / f"item={item_id}" / f"year={year}"
        index_dir.mkdir(parents=True, exist_ok=True)
        index = build_faiss_index(mat, use_gpu=cfg.faiss_use_gpu)
        faiss.write_index(_faiss_to_cpu(index), str(index_dir / "pooled.faiss"))
        (index_dir / "pooled_ids.json").write_text(json.dumps(firm_ids.tolist()), encoding="utf-8")

    items_dir = scoped_out_dir / "item_vectors"
    items_dir.mkdir(parents=True, exist_ok=True)
    items_noidx = items_df.drop(columns=["_row_idx"])
    for year in sorted(set(int(y) for y in items_noidx["year"].tolist())):
        out_path = items_dir / f"item_vectors_{year}.parquet"
        items_noidx[items_noidx["year"] == year].to_parquet(out_path, index=False)
        print(f"[DONE] Wrote: {out_path}")

    if not units_df.empty:
        units_dir = scoped_out_dir / "units"
        units_dir.mkdir(parents=True, exist_ok=True)
        for year in sorted(set(int(y) for y in units_df["year"].tolist())):
            out_path = units_dir / f"units_{year}.parquet"
            units_df[units_df["year"] == year].to_parquet(out_path, index=False)
            print(f"[DONE] Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local FAISS vector artifacts for item peer matching")
    parser.add_argument("--filing_dir", default="sec_filings", help="Root directory of extracted filing JSON")
    parser.add_argument("--out_dir", required=True, help="Output directory for vector artifacts")
    parser.add_argument("--filing", default="10-K", choices=["10-K", "10-Q"], dest="filing_type")
    parser.add_argument(
        "--year",
        "--years",
        type=int,
        nargs="+",
        default=None,
        dest="years",
        help="Optional year filter. Example: --year 2022 2023",
    )
    parser.add_argument(
        "--scope",
        default="all",
        choices=["heading", "all", "summary"],
        help="Vector scope: raw item text, extracted headings, or local summaries.",
    )
    parser.add_argument("--items", default=",".join(DEFAULT_ITEMS), help="Comma-separated item ids")
    parser.add_argument(
        "--embed_model",
        default=DEFAULT_EMBED_MODEL,
        help="Local embedding model name. Default is BAAI/bge-m3.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
        help="Tokenizer used only for chunk sizing heuristics.",
    )
    parser.add_argument(
        "--chunk_tokens",
        type=int,
        default=None,
        help="Optional global chunk size override. Default is item-specific auto sizing.",
    )
    parser.add_argument(
        "--overlap_tokens",
        type=int,
        default=None,
        help="Optional global overlap override. Default auto-scales with chunk size.",
    )
    parser.add_argument("--min_unit_tokens", type=int, default=80, help="Minimum tokens for one chunk")
    parser.add_argument("--min_units_per_item", type=int, default=3, help="Minimum chunk count for pooled items")
    parser.add_argument(
        "--faiss-gpu",
        dest="faiss_use_gpu",
        action="store_true",
        default=True,
        help="Use FAISS GPU acceleration when available (default: on).",
    )
    parser.add_argument("--no-faiss-gpu", dest="faiss_use_gpu", action="store_false")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and rebuild the target scope directory before writing outputs.",
    )
    args = parser.parse_args()

    cfg = BuildConfig(
        filing_dir=Path(args.filing_dir),
        out_dir=Path(args.out_dir),
        filing_type=str(args.filing_type),
        years=None if not args.years else sorted({int(y) for y in args.years}),
        scope=str(args.scope),
        items=[normalize_item_id(x) for x in str(args.items).split(",") if x.strip()],
        embed_model=str(args.embed_model),
        tokenizer_model=str(args.tokenizer_model),
        chunk_tokens=None if args.chunk_tokens is None else int(args.chunk_tokens),
        overlap_tokens=None if args.overlap_tokens is None else int(args.overlap_tokens),
        min_unit_tokens=int(args.min_unit_tokens),
        min_units_per_item=int(args.min_units_per_item),
        faiss_use_gpu=bool(args.faiss_use_gpu),
        overwrite=bool(args.overwrite),
    )
    build(cfg)


if __name__ == "__main__":
    main()
