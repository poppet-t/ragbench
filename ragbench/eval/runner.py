import json
import os
import time
from typing import Dict, Any, List

from ..pdf_extract import extract_pages
from ..chunking import chunk_pages
from ..embeddings import Embedder
from ..llm import LLMFactory
from ..backends.sqlite_vec import SQLiteVectorStore
from ..backends.milvus_store import MilvusVectorStore
from ..backends.memos_http import MemOSHTTPStore
from ..rag.retriever import index_backend, retrieve_with_timing
from ..rag.answerer import answer_question
from .dataset import load_qa
from ..progress import progress_iter
from .metrics import (
    exact_match,
    semantic_similarity,
    citation_hit,
    retrieval_recall_at_k,
    citation_precision,
    citation_recall,
    citation_page_distance,
    groundedness,
    numeric_fidelity,
    abstain_metrics,
)


def build_backend(
    name: str,
    out_dir: str,
    dim: int,
    memos_url: str,
    memos_user_id: str,
    memos_cube_id: str,
    memos_async_mode: str,
) -> Any:
    if name in {"sqlite", "sqlite_rag"}:
        return SQLiteVectorStore(path=os.path.join(out_dir, "sqlite_vec.db"))
    if name in {"milvus", "milvus_rag"}:
        return MilvusVectorStore(collection="ragbench", dim=dim)
    if name == "memos_rag":
        return MemOSHTTPStore(
            base_url=memos_url,
            user_id=memos_user_id,
            mem_cube_id=memos_cube_id,
            async_mode=memos_async_mode,
        )
    if name == "pageindex_sqlite":
        return SQLiteVectorStore(path=os.path.join(out_dir, "pageindex_sqlite.db"))
    if name == "pageindex_milvus":
        return MilvusVectorStore(collection="ragbench_pageindex", dim=dim)
    raise ValueError(f"Unknown backend {name}")


def run_benchmark(
    backend_name: str,
    pdfs: List[Dict[str, str]],
    qa_path: str,
    chunk_size: int,
    chunk_overlap: int,
    out_dir: str,
    mode: str,
    top_k: int = 5,
    abstain_threshold: float = 0.1,
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    llm_index_provider: str = "mock",
    llm_index_model: str | None = None,
    llm_answer_provider: str = "mock",
    llm_answer_model: str | None = None,
    force_reindex: bool = False,
    memos_url: str = "http://localhost:8000",
    memos_user_id: str = "default",
    memos_cube_id: str = "default",
    memos_async_mode: str = "sync",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    embedder = Embedder(provider=embedding_provider, model=embedding_model)
    llm_index = LLMFactory(provider=llm_index_provider, model=llm_index_model).build()
    llm_answer = LLMFactory(provider=llm_answer_provider, model=llm_answer_model).build()

    # Ingest and chunk
    chunks: List[Dict[str, Any]] = []
    for spec in pdfs:
        pages = extract_pages(spec["path"], spec["doc_id"], spec["filename"])
        chunks.extend(chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap, method="default"))

    # Index
    dim = len(embedder.embed(["test"])[0])
    embedder.embed_calls = 0
    embedder.embed_texts = 0
    backend = build_backend(
        backend_name,
        out_dir,
        dim=dim,
        memos_url=memos_url,
        memos_user_id=memos_user_id,
        memos_cube_id=memos_cube_id,
        memos_async_mode=memos_async_mode,
    )
    index_meta = _build_index_meta(
        backend_name=backend_name,
        mode=mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dim=dim,
        llm_index_provider=llm_index_provider,
        llm_index_model=llm_index_model,
        pdfs=pdfs,
    )
    meta_path = _index_meta_path(out_dir, backend_name)
    index_cached = False
    if not force_reindex:
        prev_meta = _load_index_meta(meta_path)
        if prev_meta == index_meta and _index_store_ready(backend_name, out_dir, backend):
            index_cached = True

    if index_cached:
        index_build_ms = 0.0
        indexed_items = []
        chunk_lookup = {c["chunk_id"]: c for c in chunks}
    else:
        index_start = time.perf_counter()
        card_cache_path = _index_cards_cache_path(
            out_dir,
            backend_name=backend_name,
            llm_index_provider=llm_index_provider,
            llm_index_model=llm_index_model,
        )
        indexed_items, chunk_lookup = index_backend(
            backend,
            chunks,
            embedder,
            mode=mode,
            llm=llm_index,
            card_cache_path=card_cache_path,
        )
        index_build_ms = (time.perf_counter() - index_start) * 1000
        _write_index_meta(meta_path, index_meta)
    index_size_bytes = _index_size_bytes(backend_name, out_dir)

    qa_all = load_qa(qa_path)
    doc_ids = {p["doc_id"] for p in pdfs}
    qa = [
        row
        for row in qa_all
        if not row.get("doc_id") or row["doc_id"] in doc_ids or row.get("doc_id") == "mixed"
    ]
    results: List[Dict[str, Any]] = []
    for row in progress_iter(qa, desc=f"eval {backend_name}", total=len(qa)):
        retrieved, embed_ms, search_ms = retrieve_with_timing(
            backend,
            row["question"],
            embedder,
            top_k=top_k,
            mode=mode,
            chunk_lookup=chunk_lookup,
        )
        retrieve_ms = embed_ms + search_ms
        ans = answer_question(
            row["question"],
            retrieved,
            abstain_threshold=abstain_threshold,
            llm=llm_answer,
            use_llm=llm_answer_provider != "mock",
        )
        ans["latency_ms_retrieval"] = retrieve_ms
        ans["latency_ms_embed"] = embed_ms
        ans["latency_ms_search"] = search_ms
        ans["latency_ms_total"] = ans["latency_ms_answer"] + retrieve_ms
        grounded_pass, grounded_score = groundedness(row, ans["final_answer"], retrieved)
        num_exact, num_f1 = numeric_fidelity(ans["final_answer"], row.get("ground_truth_answer", ""))
        abstain_correct, false_abstain = abstain_metrics(row, ans["final_answer"])
        cite_dist = citation_page_distance(ans["citations"], row.get("ground_truth_citations", []))
        rec = {
            "question_id": row["id"],
            "backend": backend_name,
            "doc_id": row.get("doc_id"),
            "is_negative": row.get("is_negative"),
            "hop": row.get("hop"),
            "final_answer": ans["final_answer"],
            "citations": ans["citations"],
            "rationale": ans["rationale"],
            "retrieved": retrieved,
            "metrics": {
                "retrieval_recall_at_k": retrieval_recall_at_k(retrieved, row.get("ground_truth_citations", [])),
                "citation_hit": citation_hit(ans["citations"], row.get("ground_truth_citations", [])),
                "citation_precision": citation_precision(ans["citations"], row.get("ground_truth_citations", [])),
                "citation_recall": citation_recall(ans["citations"], row.get("ground_truth_citations", [])),
                "citation_page_distance": cite_dist,
                "exact_match": exact_match(ans["final_answer"], row.get("ground_truth_answer", "")),
                "semantic_similarity": semantic_similarity(ans["final_answer"], row.get("ground_truth_answer", "")),
                "groundedness_pass": grounded_pass,
                "groundedness_score": grounded_score,
                "numeric_exact": num_exact,
                "numeric_f1": num_f1,
                "abstain_correct": abstain_correct,
                "false_abstain": false_abstain,
            },
            "latency_ms_retrieval": ans["latency_ms_retrieval"],
            "latency_ms_embed": ans["latency_ms_embed"],
            "latency_ms_search": ans["latency_ms_search"],
            "latency_ms_answer": ans["latency_ms_answer"],
            "latency_ms_total": ans["latency_ms_total"],
        }
        results.append(rec)

    out_path = os.path.join(out_dir, f"{backend_name}_results.json")
    pipeline = {
        "index_build_ms": index_build_ms,
        "index_size_bytes": index_size_bytes,
        "embed_calls": embedder.embed_calls,
        "embed_texts": embedder.embed_texts,
        "llm_index_calls": getattr(llm_index, "index_calls", 0),
        "llm_answer_calls": getattr(llm_answer, "answer_calls", 0),
        "index_cached": index_cached,
    }
    if hasattr(backend, "index_time_ms"):
        pipeline["memos_index_time_ms"] = getattr(backend, "index_time_ms", None)
        pipeline["memos_add_calls"] = getattr(backend, "add_calls", None)
        pipeline["memos_search_calls"] = getattr(backend, "search_calls", None)
    for rec in results:
        rec["pipeline"] = pipeline
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return {"results": results, "out_path": out_path}


def _index_size_bytes(backend_name: str, out_dir: str) -> int | None:
    if backend_name in {"sqlite", "sqlite_rag"}:
        path = os.path.join(out_dir, "sqlite_vec.db")
    elif backend_name == "pageindex_sqlite":
        path = os.path.join(out_dir, "pageindex_sqlite.db")
    else:
        return None
    return os.path.getsize(path) if os.path.exists(path) else None


def _index_meta_path(out_dir: str, backend_name: str) -> str:
    return os.path.join(out_dir, f"{backend_name}_index_meta.json")


def _build_index_meta(
    backend_name: str,
    mode: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_provider: str,
    embedding_model: str | None,
    embedding_dim: int,
    llm_index_provider: str,
    llm_index_model: str | None,
    pdfs: List[Dict[str, str]],
) -> Dict[str, Any]:
    pdf_meta = []
    for spec in pdfs:
        try:
            st = os.stat(spec["path"])
            size = st.st_size
            mtime = int(st.st_mtime)
        except OSError:
            size = None
            mtime = None
        pdf_meta.append(
            {
                "doc_id": spec.get("doc_id"),
                "filename": spec.get("filename"),
                "path": spec.get("path"),
                "size": size,
                "mtime": mtime,
            }
        )
    return {
        "version": 1,
        "backend": backend_name,
        "mode": mode,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "llm_index_provider": llm_index_provider,
        "llm_index_model": llm_index_model,
        "pdfs": pdf_meta,
    }


def _safe_token(value: str | None) -> str:
    token = (value or "default").strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in token)


def _index_cards_cache_path(
    out_dir: str,
    backend_name: str,
    llm_index_provider: str,
    llm_index_model: str | None,
) -> str | None:
    if not backend_name.startswith("pageindex"):
        return None
    provider = _safe_token(llm_index_provider)
    model = _safe_token(llm_index_model)
    filename = f"{backend_name}_cards_{provider}_{model}.jsonl"
    return os.path.join(out_dir, filename)


def _load_index_meta(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_index_meta(path: str, meta: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _index_store_ready(backend_name: str, out_dir: str, backend: Any) -> bool:
    if backend_name in {"sqlite", "sqlite_rag"}:
        path = os.path.join(out_dir, "sqlite_vec.db")
        return os.path.exists(path) and os.path.getsize(path) > 0
    if backend_name == "pageindex_sqlite":
        path = os.path.join(out_dir, "pageindex_sqlite.db")
        return os.path.exists(path) and os.path.getsize(path) > 0
    if backend_name in {"milvus", "milvus_rag", "pageindex_milvus"}:
        return getattr(backend, "_milvus", None) is not None
    return False
