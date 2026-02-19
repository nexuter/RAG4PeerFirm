"""Minimal Gemini client for embeddings."""

import os
from typing import List, Optional, Sequence

import time
import requests


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_EMBED_MODEL") or "gemini-embedding-001"
        self.gen_model = os.getenv("GEMINI_GEN_MODEL") or "gemini-3-flash-preview"
        self.base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        self.timeout = float(os.getenv("GEMINI_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
        self.backoff_base = float(os.getenv("GEMINI_BACKOFF_BASE", "2.0"))

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status not in (429, 500, 502, 503, 504):
                    raise
            except requests.exceptions.RequestException as exc:
                last_error = exc
            if attempt < self.max_retries:
                sleep_for = self.backoff_base * (2 ** attempt)
                time.sleep(sleep_for)
        if last_error:
            raise last_error
        raise RuntimeError("Embedding request failed after retries")

    def embed_text(self, text: str) -> List[float]:
        url = f"{self.base_url}/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }
        data = self._post_with_retry(url, payload)
        embedding = data.get("embedding", {}).get("values")
        if not embedding:
            raise ValueError("No embedding returned from Gemini API")
        return embedding

    def embed_texts(self, texts: Sequence[str], batch_size: int = 1) -> List[List[float]]:
        if batch_size <= 1:
            return [self.embed_text(text) for text in texts]

        url = f"{self.base_url}/models/{self.model}:batchEmbedContents?key={self.api_key}"
        results: List[List[float]] = []

        def chunked(seq: Sequence[str], size: int) -> List[Sequence[str]]:
            return [seq[i:i + size] for i in range(0, len(seq), size)]

        try:
            for chunk in chunked(list(texts), batch_size):
                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.model}",
                            "content": {"parts": [{"text": text}]},
                        }
                        for text in chunk
                    ]
                }
                data = self._post_with_retry(url, payload)
                embeddings = data.get("embeddings", [])
                if not embeddings:
                    raise ValueError("No embeddings returned from Gemini API batch endpoint")
                for entry in embeddings:
                    values = entry.get("values")
                    if not values:
                        raise ValueError("Missing embedding values in batch response")
                    results.append(values)
        except Exception:
            return [self.embed_text(text) for text in texts]

        return results

    def generate_text(self, prompt: str) -> str:
        url = f"{self.base_url}/models/{self.gen_model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        data = self._post_with_retry(url, payload)
        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError("No candidates returned from Gemini API")
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            raise ValueError("No content parts returned from Gemini API")
        text = parts[0].get("text")
        if not text:
            raise ValueError("No text returned from Gemini API")
        return str(text)
