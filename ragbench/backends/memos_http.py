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
                "memory_content": json.dumps(blob, ensure_ascii=True),
                "async_mode": self.async_mode,
                "source": "ragbench",
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
            "top_k": top_k,
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
    if isinstance(data.get("data"), dict):
        data = data["data"]
    candidates: List[Dict[str, Any]] = []
    for key in ["text_mem", "pref_mem", "act_mem", "para_mem"]:
        groups = data.get(key)
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            memories = group.get("memories")
            if not isinstance(memories, list):
                continue
            for mem in memories:
                if isinstance(mem, dict):
                    cand = mem.copy()
                    if "cube_id" not in cand and group.get("cube_id"):
                        cand["cube_id"] = group.get("cube_id")
                    candidates.append(cand)
                elif mem is not None:
                    candidates.append({"memory": mem, "cube_id": group.get("cube_id")})
    if candidates:
        return candidates
    for key in ["results", "items", "memories"]:
        val = data.get(key)
        if isinstance(val, list):
            return [c for c in val if isinstance(c, dict)]
    return []


def _candidate_to_chunk(cand: Dict[str, Any]) -> Dict[str, Any] | None:
    content = _extract_content(cand)
    if content is None:
        return None
    text_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=True)
    try:
        blob = json.loads(text_content)
    except json.JSONDecodeError:
        blob = {"text": text_content}

    meta = cand.get("metadata") if isinstance(cand.get("metadata"), dict) else {}
    info = meta.get("info") if isinstance(meta.get("info"), dict) else {}

    score = cand.get("score")
    if score is None and meta:
        score = meta.get("relativity")
    if score is None and meta:
        score = meta.get("score")
    if score is None and "distance" in cand:
        try:
            score = -float(cand["distance"])
        except Exception:
            score = 0.0
    if score is None:
        score = 0.0

    return {
        "chunk_id": _pick_value("chunk_id", blob, info, cand, meta),
        "doc_id": _pick_value("doc_id", blob, info, cand, meta),
        "pdf_filename": _pick_value("pdf_filename", blob, info, cand, meta),
        "page_start": _as_int(_pick_value("page_start", blob, info, cand, meta), 0),
        "page_end": _as_int(_pick_value("page_end", blob, info, cand, meta), 0),
        "text": _pick_value("text", blob, info, cand, meta) or text_content,
        "meta": blob.get("meta") or {},
        "score": float(score),
    }


def _extract_content(cand: Dict[str, Any]) -> Any:
    if "memory" in cand:
        return cand.get("memory")
    if "content" in cand:
        return cand.get("content")
    if "metadata" in cand and isinstance(cand.get("metadata"), dict):
        meta = cand["metadata"]
        if "memory" in meta:
            return meta.get("memory")
        if "content" in meta:
            return meta.get("content")
    if "message" in cand:
        msg = cand.get("message") or {}
        return msg.get("content") if isinstance(msg, dict) else msg
    if "messages" in cand and cand.get("messages"):
        msg = cand.get("messages")[-1]
        return msg.get("content") if isinstance(msg, dict) else msg
    return None


def _pick_value(key: str, *sources: Dict[str, Any]) -> Any:
    for src in sources:
        if isinstance(src, dict) and key in src and src.get(key) not in (None, ""):
            return src.get(key)
    return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
