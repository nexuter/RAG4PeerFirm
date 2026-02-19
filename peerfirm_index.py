"""Build a vector index for peer-firm search."""

import argparse
import os
from datetime import datetime
from typing import List

import numpy as np

from config import ITEMS_10K, ITEMS_10Q, SEC_FILINGS_DIR
from src.peerfirm.gemini_client import GeminiClient
from src.peerfirm.index_io import iter_indexable_items, save_index
from src.peerfirm.item_loader import load_item_text
from src.peerfirm.text_utils import truncate_text


def _items_for_filing(filing_type: str) -> List[str]:
    if filing_type == "10-K":
        return list(ITEMS_10K.keys())
    if filing_type == "10-Q":
        return list(ITEMS_10Q.keys())
    return []


def build_single_index(args: argparse.Namespace, item: str, index_dir: str) -> None:
    client = GeminiClient(model=args.model)

    texts: List[str] = []
    metadata = []

    for cik, year, filing_type, item_number, _path in iter_indexable_items(
        args.base_dir, args.year, args.filing_type, item
    ):
        text, used_path = load_item_text(args.base_dir, cik, year, filing_type, item_number)
        if not text:
            continue
        text = truncate_text(text, args.max_chars)
        texts.append(text)
        metadata.append({
            "cik": cik,
            "year": year,
            "filing_type": filing_type,
            "item": item_number,
            "item_path": used_path,
            "text_chars": len(text),
        })

    if not texts:
        print(f"No items found for item {item}. Skipping.")
        return

    embeddings = client.embed_texts(texts, batch_size=args.batch_size)

    emb_array = np.array(embeddings, dtype=np.float32)

    config = {
        "model": client.model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "base_dir": args.base_dir,
        "year": args.year,
        "filing_type": args.filing_type,
        "item": item,
        "count": len(metadata),
        "embedding_dim": emb_array.shape[1] if emb_array.ndim == 2 else None,
    }

    save_index(index_dir, emb_array, metadata, config)

    print(f"Saved {len(metadata)} embeddings to {index_dir}")


def build_index(args: argparse.Namespace) -> None:
    if args.item == "all":
        items = _items_for_filing(args.filing_type)
        if not items:
            raise RuntimeError("No items configured for the requested filing type")
        for item in items:
            item_dir = os.path.join(args.index_dir, f"item_{item}")
            build_single_index(args, item, item_dir)
        return

    build_single_index(args, args.item, args.index_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build peer-firm vector index")
    parser.add_argument("--year", required=True, help="Filing year, e.g. 2024")
    parser.add_argument("--item", default="all", help="Item number, e.g. 1C (default: all)")
    parser.add_argument("--filing", dest="filing_type", default="10-K", choices=["10-K", "10-Q"])
    parser.add_argument("--base-dir", default=SEC_FILINGS_DIR, help="Base sec_filings directory")
    parser.add_argument("--index-dir", default=os.path.join("vector_db", "peerfirm"), help="Output index directory")
    parser.add_argument("--model", default=None, help="Gemini embedding model override")
    parser.add_argument("--max-chars", type=int, default=20000, help="Max chars per document")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for embedding API calls")

    args = parser.parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
