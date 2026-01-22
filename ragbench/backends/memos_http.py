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
        timeout_s: int = 180,
        mode: str = "fast",
        add_mode: str | None = None,
        search_mode: str | None = None,
        search_memory_type: str | None = "All",
        dedup: str | None = "sim",
        search_top_k_multiplier: int = 3,
        search_top_k_floor: int = 0,
        fallback_search_mode: str | None = "fast",
        fallback_search_memory_type: str | None = "All",
        include_preference: bool = False,
        pref_top_k: int = 0,
        search_tool_memory: bool = False,
        tool_mem_top_k: int = 0,
        session_id: str = "default_session",
    ):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.mem_cube_id = mem_cube_id
        self.async_mode = async_mode
        self.timeout_s = timeout_s
        self.mode = mode
        self.add_mode = add_mode or mode
        self.search_mode = search_mode or mode
        self.search_memory_type = search_memory_type
        self.dedup = dedup
        self.search_top_k_multiplier = max(1, search_top_k_multiplier)
        self.search_top_k_floor = max(0, search_top_k_floor)
        self.fallback_search_mode = fallback_search_mode
        self.fallback_search_memory_type = fallback_search_memory_type
        self.include_preference = include_preference
        self.pref_top_k = max(0, pref_top_k)
        self.search_tool_memory = search_tool_memory
        self.tool_mem_top_k = max(0, tool_mem_top_k)
        self.session_id = session_id
        self.add_calls = 0
        self.search_calls = 0
        self.index_time_ms = 0.0

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _ensure_user(self) -> None:
        payload = {
            "user_id": self.user_id,
            "user_name": self.user_id,
            "mem_cube_id": self.mem_cube_id,
        }
        endpoints = [
            "/product/users/register",
            "/product/user/register",
            "/product/register",
            "/users/register",
        ]
        for endpoint in endpoints:
            url = f"{self.base_url}{endpoint}"
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout_s)
            except requests.RequestException:
                continue
            if resp.status_code == 404:
                continue
            if resp.status_code == 200:
                return
            if resp.status_code in (400, 409, 500):
                try:
                    data = resp.json()
                    msg = str(data.get("message", "")).lower()
                except ValueError:
                    msg = resp.text.lower()
                if "exist" in msg or "already" in msg:
                    return
            resp.raise_for_status()

    def index(self, items: List[Dict[str, Any]]) -> None:
        start = time.perf_counter()
        self._ensure_user()
        for item in items:
            meta_blob = {
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "pdf_filename": item.get("pdf_filename"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
            }
            memory_content = _format_memory(meta_blob, item.get("text", ""))
            info_payload = dict(meta_blob)
            info_payload["ragbench_meta_json"] = json.dumps(meta_blob, ensure_ascii=True)
            payload = {
                "user_id": self.user_id,
                "mem_cube_id": self.mem_cube_id,
                "readable_cube_ids": [self.mem_cube_id],
                "writable_cube_ids": [self.mem_cube_id],
                "messages": [{"role": "user", "content": memory_content}],
                "async_mode": self.async_mode,
                "mode": self.add_mode,
                "session_id": self.session_id,
                "info": info_payload,
            }
            self._post("/product/add", payload)
            self.add_calls += 1
        self.index_time_ms = (time.perf_counter() - start) * 1000

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError("MemOSHTTPStore does not support vector search. Use search_text().")

    def search_text(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        request_top_k = max(
            top_k,
            top_k * self.search_top_k_multiplier,
            self.search_top_k_floor,
        )
        payload = {
            "query": query_text,
            "user_id": self.user_id,
            "mem_cube_id": self.mem_cube_id,
            "readable_cube_ids": [self.mem_cube_id],
            "mode": self.search_mode,
            "top_k": request_top_k,
            "include_preference": self.include_preference,
            "pref_top_k": self.pref_top_k,
            "search_tool_memory": self.search_tool_memory,
            "tool_mem_top_k": self.tool_mem_top_k,
            "session_id": self.session_id,
        }
        if self.search_memory_type:
            payload["search_memory_type"] = self.search_memory_type
        if self.dedup:
            payload["dedup"] = self.dedup
        data = None
        candidates: List[Dict[str, Any]] = []
        try:
            data = self._post("/product/search", payload)
            candidates = _extract_candidates(data)
        except requests.RequestException:
            data = None
            candidates = []
        if not candidates and self.fallback_search_mode and self.fallback_search_mode != self.search_mode:
            payload["mode"] = self.fallback_search_mode
            if self.fallback_search_memory_type is None:
                payload.pop("search_memory_type", None)
            else:
                payload["search_memory_type"] = self.fallback_search_memory_type
            data = self._post("/product/search", payload)
            candidates = _extract_candidates(data)
        self.search_calls += 1
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
    for key in ["text_mem", "pref_mem", "act_mem", "para_mem", "tool_mem"]:
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
    meta = cand.get("metadata") if isinstance(cand.get("metadata"), dict) else {}
    info = {}
    if isinstance(cand.get("info"), dict):
        info.update(cand["info"])
    if isinstance(meta.get("info"), dict):
        info.update(meta["info"])

    blob = {}
    if isinstance(info.get("ragbench_meta"), dict):
        blob.update(info["ragbench_meta"])
    meta_blob, body_text = _parse_memory_text(text_content)
    if meta_blob:
        blob.update(meta_blob)
    if not blob:
        try:
            blob = json.loads(text_content)
        except json.JSONDecodeError:
            blob = {}

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
        "text": _pick_value("text", blob, info, cand, meta) or body_text or text_content,
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


def _format_memory(meta_blob: Dict[str, Any], text: str) -> str:
    meta_json = json.dumps(meta_blob, ensure_ascii=True)
    return f"RAGBENCH_META={meta_json}\nRAGBENCH_TEXT={text}"


def _parse_memory_text(text: str) -> tuple[Dict[str, Any], str]:
    marker = "RAGBENCH_META="
    start = text.find(marker)
    if start != -1:
        payload = text[start:]
        meta_line, _, rest = payload.partition("\n")
        meta_json = meta_line[len(marker):]
        try:
            meta_blob = json.loads(meta_json)
        except json.JSONDecodeError:
            meta_blob = {}
        body_text = rest
        if rest.startswith("RAGBENCH_TEXT="):
            body_text = rest[len("RAGBENCH_TEXT="):]
        return meta_blob, body_text
    return {}, text
