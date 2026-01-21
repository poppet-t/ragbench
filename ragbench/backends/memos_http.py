import json
import time
from typing import List, Dict, Any

import requests

from .base import VectorStore


class MemOSHTTPStore(VectorStore):
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        user_id: str = "default",
        mem_cube_id: str = "default",
        async_mode: str = "sync",
        timeout_s: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.mem_cube_id = mem_cube_id
        self.async_mode = async_mode
        self.timeout_s = timeout_s
        self.add_calls = 0
        self.search_calls = 0
        self.index_time_ms = 0.0

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def index(self, items: List[Dict[str, Any]]) -> None:
        start = time.perf_counter()
        for item in items:
            blob = {
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "pdf_filename": item.get("pdf_filename"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "text": item.get("text", ""),
                "meta": item.get("meta") or {},
            }
            payload = {
                "user_id": self.user_id,
                "mem_cube_id": self.mem_cube_id,
                "messages": [{"role": "user", "content": json.dumps(blob, ensure_ascii=True)}],
                "async_mode": self.async_mode,
            }
            self._post("/product/add", payload)
            self.add_calls += 1
        self.index_time_ms = (time.perf_counter() - start) * 1000

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError("MemOSHTTPStore does not support vector search. Use search_text().")

    def search_text(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        payload = {
            "query": query_text,
            "user_id": self.user_id,
            "mem_cube_id": self.mem_cube_id,
        }
        data = self._post("/product/search", payload)
        self.search_calls += 1
        candidates = _extract_candidates(data)
        hits: List[Dict[str, Any]] = []
        for cand in candidates:
            hit = _candidate_to_chunk(cand)
            if hit:
                hits.append(hit)
        if top_k:
            hits = hits[:top_k]
        return hits


def _extract_candidates(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    if not isinstance(data, dict):
        return []
    for key in ["results", "data", "items", "memories"]:
        val = data.get(key)
        if isinstance(val, list):
            return [c for c in val if isinstance(c, dict)]
    return []


def _candidate_to_chunk(cand: Dict[str, Any]) -> Dict[str, Any] | None:
    content = None
    if "content" in cand:
        content = cand.get("content")
    elif "message" in cand:
        msg = cand.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else msg
    elif "messages" in cand and cand.get("messages"):
        msg = cand.get("messages")[-1]
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = msg
    if content is None:
        return None

    text_content = content if isinstance(content, str) else json.dumps(content)
    try:
        blob = json.loads(text_content)
    except json.JSONDecodeError:
        blob = {"text": text_content}

    score = cand.get("score")
    if score is None and "distance" in cand:
        try:
            score = -float(cand["distance"])
        except Exception:
            score = 0.0
    if score is None:
        score = 0.0

    return {
        "chunk_id": blob.get("chunk_id") or cand.get("chunk_id"),
        "doc_id": blob.get("doc_id") or cand.get("doc_id"),
        "pdf_filename": blob.get("pdf_filename") or cand.get("pdf_filename"),
        "page_start": blob.get("page_start") or cand.get("page_start") or 0,
        "page_end": blob.get("page_end") or cand.get("page_end") or 0,
        "text": blob.get("text", ""),
        "meta": blob.get("meta") or {},
        "score": float(score),
    }
