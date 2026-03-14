"""
summrizer.py

Create local LLM summaries for filing items and save sibling `_summ.json` files
 beside each `*_item.json` source.

The summarization prompt emphasizes signals that are useful for vector search and
 peer matching: differentiated strategy, risk posture, operating focus, major
 changes from the previous year, and non-generic firm-specific facts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


DEFAULT_ITEMS = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16",
]


class TokenCounter:
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


def extract_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def infer_firm_year_from_path(path: Path, filing_dir: Path, filing_type: str) -> Tuple[str, int]:
    rel = path.relative_to(filing_dir)
    parts = rel.parts
    if len(parts) >= 4 and parts[1].isdigit() and parts[2].upper() == filing_type.upper():
        return parts[0], int(parts[1])
    raise ValueError(f"Unable to infer firm/year from path: {path}")


def list_item_files(filing_dir: Path, filing_type: str) -> List[Path]:
    files: List[Path] = []
    for path in filing_dir.rglob("*_item.json"):
        if path.is_file() and filing_type.upper() in str(path).upper() and not path.name.endswith("_item_summ.json"):
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
        value = items.get(str(key)) or items.get(item_id)
        if not isinstance(value, dict):
            continue
        text = str(value.get("text_content") or value.get("html_content") or "").strip()
        text = extract_text(text)
        if text:
            out[item_id] = text
    return out


def recommended_input_tokens(model_name: str) -> int:
    model = model_name.lower()
    if "8b" in model:
        return 6000
    if "70b" in model:
        return 8000
    if "3b" in model:
        return 3500
    return 4000


def chunk_text(words: List[str], chunk_tokens: int, overlap_tokens: int) -> List[str]:
    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(words), step):
        part = words[start : start + chunk_tokens]
        if part:
            chunks.append(" ".join(part))
    return chunks


def trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def build_map_prompt(
    item_id: str,
    target_words: int,
    current_text: str,
    previous_text: str,
) -> str:
    return f"""You are preparing a retrieval-oriented summary for SEC filing item {item_id}.

Write a dense summary that preserves facts useful for matching this firm to peers.
Focus on:
- business model, products, markets, geographies, and customer mix
- strategic changes, restructurings, acquisitions, divestitures, and capital allocation
- risk exposures, dependencies, regulation, and supply-chain specifics
- operating or financial themes that make this firm distinctive
- explicit changes versus the previous year when evidence is available

Avoid generic compliance wording, disclaimers, and repeated boilerplate.
Use firm-specific facts and terminology. Do not invent missing comparisons.
Target about {target_words} words.

Previous-year context:
{previous_text or "No prior-year item text available."}

Current text:
{current_text}
"""


def build_reduce_prompt(item_id: str, target_words: int, partial_summaries: List[str]) -> str:
    joined = "\n\n".join(f"Partial summary {idx}:\n{text}" for idx, text in enumerate(partial_summaries, start=1))
    return f"""You are merging partial summaries for SEC filing item {item_id}.

Produce one final summary optimized for vector similarity and peer-firm comparison.
Preserve the most distinctive strategic, operational, market, and risk information.
When partial summaries conflict, prefer specific and recent statements.
Avoid repetition and generic filler.
Target about {target_words} words.

{joined}
"""


def ollama_generate(base_url: str, model: str, prompt: str, timeout_sec: int) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        },
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    payload = resp.json()
    text = str(payload.get("response") or "").strip()
    if not text:
        raise RuntimeError("Empty response from local LLM.")
    return text


@dataclass
class SummarizerConfig:
    filing_dir: Path
    filing_type: str
    years: Optional[List[int]]
    items: List[str]
    model: str
    target_words: int
    ollama_url: str
    timeout_sec: int
    map_reduce: bool
    tokenizer_model: str
    overwrite: bool


def summarize_one_item(
    model: str,
    base_url: str,
    timeout_sec: int,
    token_counter: TokenCounter,
    item_id: str,
    current_text: str,
    previous_text: str,
    target_words: int,
    map_reduce: bool,
) -> str:
    max_input_tokens = recommended_input_tokens(model)
    current_tokens = token_counter.count(current_text)
    if not map_reduce or current_tokens <= max_input_tokens:
        prompt = build_map_prompt(
            item_id=item_id,
            target_words=target_words,
            current_text=current_text,
            previous_text=trim_words(previous_text, 700) if previous_text else "",
        )
        return ollama_generate(base_url, model, prompt, timeout_sec)

    overlap = max(150, int(max_input_tokens * 0.1))
    chunks = chunk_text(current_text.split(), max_input_tokens, overlap)
    prev_context = trim_words(previous_text, 700) if previous_text else ""
    partial_target = max(120, min(300, math.ceil(target_words / max(1, len(chunks)))))
    partials = [
        ollama_generate(
            base_url,
            model,
            build_map_prompt(item_id, partial_target, chunk, prev_context),
            timeout_sec,
        )
        for chunk in chunks
    ]
    reduce_prompt = build_reduce_prompt(item_id, target_words, partials)
    return ollama_generate(base_url, model, reduce_prompt, timeout_sec)


def build_summary_path(item_path: Path) -> Path:
    return item_path.with_name(item_path.stem + "_summ.json")


def run(cfg: SummarizerConfig) -> None:
    token_counter = TokenCounter(cfg.tokenizer_model)
    files = list_item_files(cfg.filing_dir, cfg.filing_type)
    if cfg.years:
        selected = set(cfg.years)
        files = [
            path
            for path in files
            if infer_firm_year_from_path(path, cfg.filing_dir, cfg.filing_type)[1] in selected
        ]
    if not files:
        raise RuntimeError("No item JSON files found for summarization.")

    for idx, path in enumerate(files, start=1):
        firm_id, year = infer_firm_year_from_path(path, cfg.filing_dir, cfg.filing_type)
        summary_path = build_summary_path(path)
        if summary_path.exists() and not cfg.overwrite:
            continue

        current_items = load_items_from_json(path, cfg.items)
        if not current_items:
            continue

        previous_items: Dict[str, str] = {}
        prev_path = path.parents[2] / str(year - 1) / cfg.filing_type / path.name
        if prev_path.exists():
            previous_items = load_items_from_json(prev_path, cfg.items)

        out_items: Dict[str, Dict[str, object]] = {}
        for item_id, current_text in current_items.items():
            previous_text = previous_items.get(item_id, "")
            summary = summarize_one_item(
                model=cfg.model,
                base_url=cfg.ollama_url,
                timeout_sec=cfg.timeout_sec,
                token_counter=token_counter,
                item_id=item_id,
                current_text=current_text,
                previous_text=previous_text,
                target_words=cfg.target_words,
                map_reduce=cfg.map_reduce,
            )
            out_items[item_id] = {
                "summary": summary.strip(),
                "target_words": cfg.target_words,
                "llm": cfg.model,
                "source_path": str(path),
                "previous_year_available": bool(previous_text),
            }

        if out_items:
            summary_path.write_text(
                json.dumps(
                    {
                        "firm_id": str(firm_id),
                        "year": int(year),
                        "filing_type": cfg.filing_type,
                        "source_path": str(path),
                        "items": out_items,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[DONE] Wrote: {summary_path}")

        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(files)} filings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize filing items into sibling _summ.json artifacts")
    parser.add_argument("--filing_dir", default="sec_filings", help="Root directory of extracted filing JSON")
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
    parser.add_argument("--items", default=",".join(DEFAULT_ITEMS), help="Comma-separated item ids")
    parser.add_argument("--llm", default="llama3.2:3b", help="Local Ollama model name")
    parser.add_argument("--len", type=int, default=1000, dest="target_words", help="Target summary length in words")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL for local Ollama")
    parser.add_argument("--timeout", type=int, default=300, dest="timeout_sec", help="LLM timeout in seconds")
    parser.add_argument(
        "--map-reduce",
        action="store_true",
        default=True,
        help="Use map-reduce when inputs exceed model-friendly size (default: on).",
    )
    parser.add_argument("--no-map-reduce", action="store_false", dest="map_reduce")
    parser.add_argument("--tokenizer-model", default="gpt-4o-mini", help="Tokenizer for chunk sizing heuristics")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing _summ.json files")
    args = parser.parse_args()

    cfg = SummarizerConfig(
        filing_dir=Path(args.filing_dir),
        filing_type=str(args.filing_type),
        years=None if not args.years else sorted({int(y) for y in args.years}),
        items=[normalize_item_id(x) for x in str(args.items).split(",") if x.strip()],
        model=str(args.llm),
        target_words=int(args.target_words),
        ollama_url=str(args.ollama_url),
        timeout_sec=int(args.timeout_sec),
        map_reduce=bool(args.map_reduce),
        tokenizer_model=str(args.tokenizer_model),
        overwrite=bool(args.overwrite),
    )
    run(cfg)


if __name__ == "__main__":
    main()
