"""Index read/write helpers."""

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def save_index(index_dir: str, embeddings: np.ndarray, metadata: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    meta_path = os.path.join(index_dir, "index.jsonl")
    with open(meta_path, "w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    emb_path = os.path.join(index_dir, "embeddings.npy")
    np.save(emb_path, embeddings)

    config_path = os.path.join(index_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def load_index(index_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    emb_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "index.jsonl")
    config_path = os.path.join(index_dir, "config.json")

    if not os.path.isfile(emb_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError("Index files not found in index_dir")

    embeddings = np.load(emb_path)

    metadata: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    config: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    return embeddings, metadata, config


def iter_indexable_items(base_dir: str, year: str, filing_type: str, item: str) -> Iterable[Tuple[str, str, str, str, str]]:
    for cik in os.listdir(base_dir):
        cik_path = os.path.join(base_dir, cik)
        if not os.path.isdir(cik_path):
            continue
        item_path = os.path.join(base_dir, cik, year, filing_type, "items")
        if not os.path.isdir(item_path):
            continue
        filename = f"{cik}_{year}_{filing_type}_item{item}.json"
        full_path = os.path.join(item_path, filename)
        if os.path.isfile(full_path):
            yield cik, year, filing_type, item, full_path
        else:
            alt_name = f"{cik}_{year}_{filing_type}_item{item}_xtr.json"
            alt_path = os.path.join(item_path, alt_name)
            if os.path.isfile(alt_path):
                yield cik, year, filing_type, item, alt_path
