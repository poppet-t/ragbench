import argparse
import os

from ragbench.config import PDFSpec
from ragbench.pdf_extract import extract_pages
from ragbench.chunking import chunk_pages
from ragbench.embeddings import Embedder
from ragbench.backends.sqlite_vec import SQLiteVectorStore
from ragbench.backends.milvus_store import MilvusVectorStore
from ragbench.pageindex.index_cards import build_index_cards
from ragbench.rag.retriever import build_embeddings
from ragbench.llm import LLMFactory


def main():
    parser = argparse.ArgumentParser(description="Build index for RAGBench")
    parser.add_argument("--backend", required=True, choices=["sqlite_rag", "milvus_rag", "pageindex_sqlite", "pageindex_milvus", "sqlite", "milvus"])
    parser.add_argument("--pdfs", nargs="+", required=True, help="List of pdf paths")
    parser.add_argument("--doc-ids", nargs="+", required=True, help="List of doc ids matching pdfs")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--embedding-provider", default="hash")
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdfs = [PDFSpec(doc_id=d, path=p, filename=os.path.basename(p)) for d, p in zip(args.doc_ids, args.pdfs)]

    chunks = []
    for spec in pdfs:
        pages = extract_pages(spec.path, spec.doc_id, spec.filename)
        chunks.extend(chunk_pages(pages, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap))

    embedder = Embedder(provider=args.embedding_provider, model=args.embedding_model)
    llm = LLMFactory().build()

    if args.backend == "sqlite":
        args.backend = "sqlite_rag"
    if args.backend == "milvus":
        args.backend = "milvus_rag"

    if args.backend == "sqlite_rag":
        store = SQLiteVectorStore(path=os.path.join(args.out_dir, "sqlite_vec.db"))
        items = build_embeddings(chunks, embedder)
        store.index(items)
    elif args.backend == "milvus_rag":
        store = MilvusVectorStore(collection="ragbench", dim=len(embedder.embed(['test'])[0]))
        items = build_embeddings(chunks, embedder)
        store.index(items)
    elif args.backend == "pageindex_sqlite":
        store = SQLiteVectorStore(path=os.path.join(args.out_dir, "pageindex_sqlite.db"))
        cards = build_index_cards(chunks, llm, embedder)
        store.index(cards)
    elif args.backend == "pageindex_milvus":
        store = MilvusVectorStore(collection="ragbench_pageindex", dim=len(embedder.embed(['test'])[0]))
        cards = build_index_cards(chunks, llm, embedder)
        store.index(cards)
    else:
        raise ValueError("Unknown backend")

    print(f"Indexed {len(chunks)} chunks using backend {args.backend}")


if __name__ == "__main__":
    main()
