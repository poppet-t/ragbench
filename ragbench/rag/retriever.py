import time
from typing import List, Dict, Any, Tuple

from ..embeddings import Embedder
from ..backends.base import VectorStore
from ..pageindex.index_cards import build_index_cards, pageindex_retrieve


def build_embeddings(chunks: List[Dict[str, Any]], embedder: Embedder) -> List[Dict[str, Any]]:
    vectors = embedder.embed([c["text"] for c in chunks])
    with_vectors: List[Dict[str, Any]] = []
    for chunk, vec in zip(chunks, vectors):
        item = chunk.copy()
        item["embedding"] = vec
        with_vectors.append(item)
    return with_vectors


def index_backend(
    backend: VectorStore,
    chunks: List[Dict[str, Any]],
    embedder: Embedder,
    mode: str = "chunk_vectors",
    llm=None,
    card_cache_path: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Index chunks. Returns (indexed_items, chunk_lookup)."""
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    if mode == "index_cards":
        if llm is None:
            raise ValueError("LLM required for index_cards mode")
        cards = build_index_cards(chunks, llm, embedder, cache_path=card_cache_path)
        backend.index(cards)
        return cards, chunk_lookup
    items = build_embeddings(chunks, embedder)
    backend.index(items)
    return items, chunk_lookup


def retrieve(
    backend: VectorStore,
    query: str,
    embedder: Embedder,
    top_k: int,
    mode: str,
    chunk_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if mode == "index_cards":
        q_vec = embedder.embed([query])[0]
        return pageindex_retrieve(backend, q_vec, top_k, chunk_lookup)
    if hasattr(backend, "search_text"):
        return backend.search_text(query, top_k)
    q_vec = embedder.embed([query])[0]
    hits = backend.search(q_vec, top_k)
    enriched: List[Dict[str, Any]] = []
    for hit in hits:
        chunk_id = hit.get("chunk_id")
        if chunk_id and chunk_id in chunk_lookup:
            base = chunk_lookup[chunk_id].copy()
            base["score"] = hit.get("score", 0.0)
            enriched.append(base)
        else:
            enriched.append(hit)
    return enriched


def retrieve_with_timing(
    backend: VectorStore,
    query: str,
    embedder: Embedder,
    top_k: int,
    mode: str,
    chunk_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float, float]:
    search_start = time.perf_counter()
    if mode == "index_cards":
        embed_start = time.perf_counter()
        q_vec = embedder.embed([query])[0]
        embed_ms = (time.perf_counter() - embed_start) * 1000
        retrieved = pageindex_retrieve(backend, q_vec, top_k, chunk_lookup)
    elif hasattr(backend, "search_text"):
        retrieved = backend.search_text(query, top_k)
        embed_ms = 0.0
    else:
        embed_start = time.perf_counter()
        q_vec = embedder.embed([query])[0]
        embed_ms = (time.perf_counter() - embed_start) * 1000
        hits = backend.search(q_vec, top_k)
        retrieved = []
        for hit in hits:
            chunk_id = hit.get("chunk_id")
            if chunk_id and chunk_id in chunk_lookup:
                base = chunk_lookup[chunk_id].copy()
                base["score"] = hit.get("score", 0.0)
                retrieved.append(base)
            else:
                retrieved.append(hit)
    search_ms = (time.perf_counter() - search_start) * 1000
    return retrieved, embed_ms, search_ms
