import argparse
import json
import os

from ragbench.eval.runner import run_benchmark
from ragbench.eval.report import write_report


def main():
    parser = argparse.ArgumentParser(description="Run RAGBench evaluation")
    parser.add_argument(
        "--backend",
        nargs="+",
        required=True,
        choices=["sqlite_rag", "milvus_rag", "pageindex_sqlite", "pageindex_milvus", "memos_rag", "sqlite", "milvus"],
        help="One or more backends to evaluate",
    )
    parser.add_argument("--pdfs", nargs="+", required=True, help="List of pdf paths")
    parser.add_argument("--doc-ids", nargs="+", required=True, help="List of doc ids matching pdfs")
    parser.add_argument("--qa", default="data/qa_seed.jsonl")
    parser.add_argument("--qa-set", choices=["full", "reduced"], default=None)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embedding-provider", default="hash")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--llm-provider", default="mock")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-index-provider", default=None)
    parser.add_argument("--llm-index-model", default=None)
    parser.add_argument("--llm-answer-provider", default=None)
    parser.add_argument("--llm-answer-model", default=None)
    parser.add_argument("--abstain-threshold", type=float, default=0.1)
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild index even if cache matches")
    parser.add_argument("--memos-url", default="http://localhost:8000")
    parser.add_argument("--memos-user-id", default="default")
    parser.add_argument("--memos-cube-id", default="default")
    parser.add_argument("--memos-async-mode", default="sync")
    args = parser.parse_args()

    if args.qa_set:
        args.qa = "data/qa_seed.jsonl" if args.qa_set == "full" else "data/qa_reduced_25.jsonl"

    pdf_specs = [{"doc_id": d, "path": p, "filename": os.path.basename(p)} for d, p in zip(args.doc_ids, args.pdfs)]
    all_results = []
    per_backend_root = os.path.join(args.out_dir, "individual_results")
    os.makedirs(per_backend_root, exist_ok=True)
    for backend in args.backend:
        if backend == "sqlite":
            backend = "sqlite_rag"
        if backend == "milvus":
            backend = "milvus_rag"
        mode = "index_cards" if "pageindex" in backend else "chunk_vectors"
        llm_index_provider = args.llm_index_provider or args.llm_provider
        llm_index_model = args.llm_index_model or args.llm_model
        llm_answer_provider = args.llm_answer_provider or args.llm_provider
        llm_answer_model = args.llm_answer_model or args.llm_model
        res = run_benchmark(
            backend_name=backend,
            pdfs=pdf_specs,
            qa_path=args.qa,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            out_dir=args.out_dir,
            mode=mode,
            top_k=args.k,
            abstain_threshold=args.abstain_threshold,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            llm_index_provider=llm_index_provider,
            llm_index_model=llm_index_model,
            llm_answer_provider=llm_answer_provider,
            llm_answer_model=llm_answer_model,
            force_reindex=args.force_reindex,
            memos_url=args.memos_url,
            memos_user_id=args.memos_user_id,
            memos_cube_id=args.memos_cube_id,
            memos_async_mode=args.memos_async_mode,
        )
        all_results.extend(res["results"])
        per_backend_dir = os.path.join(per_backend_root, backend)
        os.makedirs(per_backend_dir, exist_ok=True)
        with open(os.path.join(per_backend_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res["results"], f, ensure_ascii=False, indent=2)
        write_report(res["results"], per_backend_dir)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    write_report(all_results, args.out_dir)
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
