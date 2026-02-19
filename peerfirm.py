"""Query peer-firm similarity using a local index."""

import argparse
import csv
import json
import os
from typing import Iterable, List, Optional

import numpy as np

from config import SEC_FILINGS_DIR
from src.peerfirm.gemini_client import GeminiClient
from src.peerfirm.index_io import load_index
from src.peerfirm.item_loader import load_item_text
from src.peerfirm.similarity import cosine_scores, top_k
from src.peerfirm.text_utils import truncate_text


def build_prompt(query_cik: str, year: str, item: str, keywords: Optional[str], results: List[dict]) -> str:
    parts = [
        "You are a financial analyst.",
        f"Find the top {len(results)} peer companies similar to CIK {query_cik} for Item {item} in {year}.",
    ]
    if keywords:
        parts.append(f"Focus on these keywords/themes: {keywords}.")
    parts.append("Candidates:")
    for rank, row in enumerate(results, start=1):
        parts.append(f"{rank}. CIK {row['cik']} (score {row['score']:.4f})")
    parts.append("Return a ranked list with brief reasoning.")
    return "\n".join(parts)

def build_headings_prompt(
    query_cik: str,
    year: str,
    item: str,
    keywords: Optional[str],
    query_headings: List[str],
    query_bodies: Optional[List[str]] = None,
) -> str:
    parts = [
        "You are a financial analyst.",
        f"Find the top peer companies similar to CIK {query_cik} for Item {item} in {year}.",
        "Use only the section headings and bodies listed below.",
    ]
    if keywords:
        parts.append(f"Focus on these keywords/themes: {keywords}.")
    if query_headings:
        parts.append(f"Target (CIK {query_cik}) headings:")
        for heading in query_headings:
            parts.append(f"- {heading}")
    if query_bodies:
        parts.append(f"Target (CIK {query_cik}) bodies:")
        for body in query_bodies:
            parts.append(f"- {body}")
    parts.append("Return a ranked list of top peers with brief reasoning.")
    return "\n".join(parts)


def _item_json_path(base_dir: str, cik: str, year: str, filing_type: str, item: str) -> str:
    filename = f"{cik}_{year}_{filing_type}_item{item}.json"
    return os.path.join(base_dir, cik, year, filing_type, "items", filename)


def _load_structure(path: str) -> Optional[List[dict]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    structure = payload.get("structure")
    if not isinstance(structure, list):
        return None
    return structure


def _structure_path(base_dir: str, cik: str, year: str, filing_type: str, item: str) -> str:
    filename = f"{cik}_{year}_{filing_type}_item{item}_xtr.json"
    return os.path.join(base_dir, cik, year, filing_type, "items", filename)


def _flatten_headings(nodes: List[dict]) -> Iterable[str]:
    for node in nodes:
        heading = node.get("heading")
        if heading:
            yield str(heading).strip()
        children = node.get("children") or []
        if children:
            yield from _flatten_headings(children)


def _clean_heading(text: str, max_chars: int = 200) -> str:
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _flatten_bodies(nodes: List[dict]) -> Iterable[str]:
    for node in nodes:
        body = node.get("body")
        if body:
            yield str(body).strip()
        children = node.get("children") or []
        if children:
            yield from _flatten_bodies(children)


def write_csv(output_path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["rank", "cik", "year", "filing_type", "item", "score", "item_path"]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_prompt(output_path: str, prompt: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)

def write_response(output_path: str, response: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(response)

def main() -> None:
    parser = argparse.ArgumentParser(description="Find peer companies using embeddings")
    parser.add_argument("--k", type=int, default=5, help="Top-k peers to return")
    parser.add_argument("--cik", required=True, help="Target CIK")
    parser.add_argument("--year", required=True, help="Filing year")
    parser.add_argument("--item", required=True, help="Item number, e.g. 1C")
    parser.add_argument("--filing", dest="filing_type", default="10-K", choices=["10-K", "10-Q"])
    parser.add_argument("--base-dir", default=SEC_FILINGS_DIR, help="Base sec_filings directory")
    parser.add_argument("--index-dir", default=os.path.join("vector_db", "peerfirm"), help="Index directory")
    parser.add_argument("--output", default=os.path.join("output", "peerfirm"), help="Output directory")
    parser.add_argument("--keywords", default=None, help="Optional keywords to include in prompt")
    parser.add_argument("--save-prompt", action="store_true", help="Write prompt text file")
    parser.add_argument("--max-chars", type=int, default=20000, help="Max chars per document")
    parser.add_argument("--include-self", action="store_true", help="Include the target CIK in results")
    parser.add_argument("--model", default=None, help="Gemini embedding model override")
    parser.add_argument("--no-generate", action="store_false", dest="generate", help="Skip Gemini generation call")
    parser.add_argument(
        "--method",
        default="vdb",
        choices=["head", "headbody", "vdb"],
        help="Peer-finding method: head, headbody, or vdb",
    )
    parser.add_argument("--max-headings", type=int, default=30, help="Max headings for the target company")
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=0,
        help="Max chars per body entry in headbody mode (0 = no limit)",
    )

    args = parser.parse_args()

    if args.method in ("head", "headbody"):
        query_headings: List[str] = []
        query_bodies: List[str] = []

        query_structure_path = _structure_path(args.base_dir, args.cik, args.year, args.filing_type, args.item)
        if os.path.isfile(query_structure_path):
            structure = _load_structure(query_structure_path)
            if structure:
                query_headings = [
                    _clean_heading(h) for h in _flatten_headings(structure)
                ][: args.max_headings]
                if args.method == "headbody":
                    raw_bodies = [b for b in _flatten_bodies(structure) if b and b.strip()]
                    if args.max_body_chars and args.max_body_chars > 0:
                        query_bodies = [
                            _clean_heading(b, max_chars=args.max_body_chars)
                            for b in raw_bodies
                        ][: args.max_headings]
                    else:
                        query_bodies = raw_bodies[: args.max_headings]

        if args.method == "head":
            if not query_headings:
                raise RuntimeError("No headings found for the target CIK/year/item")
        else:
            if not query_headings and not query_bodies:
                raise RuntimeError("No headings/bodies found for the target CIK/year/item")

        prompt = build_headings_prompt(
            args.cik,
            args.year,
            args.item,
            args.keywords,
            query_headings,
            query_bodies if args.method == "headbody" else None,
        )
        suffix = "headings" if args.method == "head" else "headings_bodies"
        output_name = f"{args.cik}_{args.year}_{args.item}_{args.k}_{suffix}_prompt.txt"
        output_path = os.path.join(args.output, output_name)
        write_prompt(output_path, prompt)
        print(f"Wrote {args.method} prompt to {output_path}")

        if args.generate:
            client = GeminiClient()
            response_text = client.generate_text(prompt)
            response_name = f"{args.cik}_{args.year}_{args.item}_{args.k}_{suffix}_response.txt"
            response_path = os.path.join(args.output, response_name)
            write_response(response_path, response_text)
            print(f"Wrote {args.method} response to {response_path}")
        return

    query_text, _ = load_item_text(args.base_dir, args.cik, args.year, args.filing_type, args.item)
    if not query_text:
        raise RuntimeError("Target item text not found")

    query_text = truncate_text(query_text, args.max_chars)

    embeddings, metadata, _config = load_index(args.index_dir)
    if embeddings.size == 0:
        raise RuntimeError("Index has no embeddings")

    client = GeminiClient(model=args.model)
    query_embedding = np.array(client.embed_text(query_text), dtype=np.float32)

    scores = cosine_scores(query_embedding, embeddings)

    results = []
    for idx, score in enumerate(scores):
        row = metadata[idx]
        if not args.include_self and row.get("cik") == args.cik:
            continue
        results.append({
            "cik": row.get("cik"),
            "year": row.get("year"),
            "filing_type": row.get("filing_type"),
            "item": row.get("item"),
            "score": float(score),
            "item_path": row.get("item_path"),
        })

    if not results:
        raise RuntimeError("No peer results found (after filtering)")

    scores_array = np.array([row["score"] for row in results], dtype=np.float32)
    top_idx, top_scores = top_k(scores_array, args.k)

    top_rows = []
    for rank, local_idx in enumerate(top_idx, start=1):
        row = results[int(local_idx)]
        row["rank"] = rank
        row["score"] = float(top_scores[rank - 1])
        top_rows.append(row)

    output_name = f"{args.cik}_{args.year}_{args.item}_{args.k}.csv"
    output_path = os.path.join(args.output, output_name)
    write_csv(output_path, top_rows)

    prompt = build_prompt(args.cik, args.year, args.item, args.keywords, top_rows)
    if args.save_prompt:
        prompt_name = f"{args.cik}_{args.year}_{args.item}_{args.k}_prompt.txt"
        prompt_path = os.path.join(args.output, prompt_name)
        write_prompt(prompt_path, prompt)

    if args.generate:
        response_text = client.generate_text(prompt)
        response_name = f"{args.cik}_{args.year}_{args.item}_{args.k}_response.txt"
        response_path = os.path.join(args.output, response_name)
        write_response(response_path, response_text)

    print(f"Wrote {len(top_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
