from dataclasses import dataclass
import hashlib
import json
import os
from typing import List, Dict, Any, Optional

from ..embeddings import Embedder
from ..progress import progress_iter


@dataclass
class IndexCard:
    card_id: str
    source_chunk_id: str
    doc_id: str
    pdf_filename: str
    page_start: int
    page_end: int
    card_text: str
    embedding_text: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.card_id,
            "card_id": self.card_id,
            "source_chunk_id": self.source_chunk_id,
            "doc_id": self.doc_id,
            "pdf_filename": self.pdf_filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.card_text,
            "embedding_text": self.embedding_text,
            "meta": self.meta,
        }


def _build_card_text(card: Dict[str, Any]) -> str:
    title = card.get("title_guess") or "Index Card"
    facts = card.get("key_facts") or []
    entities = card.get("entities") or []
    suggested = card.get("suggested_questions") or []
    lines = [f"Title: {title}"]
    if facts:
        lines.append("Key facts: " + "; ".join(facts[:3]))
    if entities:
        lines.append("Entities: " + ", ".join(entities[:8]))
    if suggested:
        lines.append("Suggested Qs: " + "; ".join(suggested[:3]))
    return "\n".join(lines)


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path or not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = row.get("chunk_id")
            if chunk_id:
                cache[chunk_id] = row
    return cache


def _write_cache(cache_path: str, cache: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for row in cache.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_card(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title_guess": raw.get("title_guess") or "Index Card",
        "key_facts": raw.get("key_facts") or [],
        "entities": raw.get("entities") or [],
        "suggested_questions": raw.get("suggested_questions") or [],
        "embedding_text": raw.get("embedding_text") or "",
    }


def build_index_cards(
    chunks: List[Dict[str, Any]],
    llm,
    embedder: Embedder,
    cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cache = _load_cache(cache_path) if cache_path else {}
    updated_cache = dict(cache)
    cache_hits = 0
    cards: List[IndexCard] = []
    for chunk in progress_iter(chunks, desc="index cards", total=len(chunks)):
        text = chunk.get("text", "") or ""
        text_hash = _text_hash(text)
        cached = cache.get(chunk["chunk_id"])
        if cached and cached.get("text_hash") == text_hash:
            raw = cached.get("card") or {}
            cache_hits += 1
        else:
            raw = llm.generate_index_card(chunk)
            updated_cache[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                "text_hash": text_hash,
                "card": _normalize_card(raw),
            }
        raw = _normalize_card(raw)
        card_text = _build_card_text(raw)
        card_id = f"{chunk['chunk_id']}_card"
        cards.append(
            IndexCard(
                card_id=card_id,
                source_chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                pdf_filename=chunk.get("pdf_filename", ""),
                page_start=chunk.get("page_start", 0),
                page_end=chunk.get("page_end", 0),
                card_text=card_text,
                embedding_text=raw.get("embedding_text") or card_text,
                meta={"index_card": True, "source_chunk_id": chunk["chunk_id"], "card_id": card_id},
            )
        )
    if cache_path and updated_cache:
        _write_cache(cache_path, updated_cache)
    embeddings = embedder.embed([c.embedding_text for c in cards])
    out: List[Dict[str, Any]] = []
    for card, emb in zip(cards, embeddings):
        row = card.to_dict()
        row["embedding"] = emb
        out.append(row)
    return out


def pageindex_retrieve(store, query_vector: List[float], top_k: int, chunk_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits = store.search(query_vector, top_k)
    contexts: List[Dict[str, Any]] = []
    seen = set()
    for hit in hits:
        source_id = hit.get("source_chunk_id")
        if not source_id:
            meta = hit.get("meta") or {}
            source_id = meta.get("source_chunk_id")
        if not source_id:
            chunk_id = hit.get("chunk_id", "")
            source_id = chunk_id[:-5] if chunk_id.endswith("_card") else chunk_id
        if not source_id or source_id in seen:
            continue
        orig = chunk_lookup.get(source_id)
        if not orig:
            continue
        seen.add(source_id)
        contexts.append(orig | {"score": hit.get("score", 0.0)})
    return contexts
