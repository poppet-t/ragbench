import json
import os
from pathlib import Path

import requests

from ragbench.eval.runner import run_benchmark
from ragbench.eval.report import write_report


def main():
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    qa_path = Path(out_dir) / "smoke_qa.jsonl"
    with open("data/qa_seed.jsonl", "r", encoding="utf-8") as f_in:
        lines = [line for line in f_in if line.strip()][:3]
    qa_path.write_text("".join(lines), encoding="utf-8")

    pdfs = [
        {"doc_id": "whitepaper", "path": "/Users/CJ/Documents/work/zCloak_AI_Whitepaper_v1.4.pdf", "filename": "zCloak_AI_Whitepaper_v1.4.pdf"},
    ]
    res = run_benchmark(
        backend_name="sqlite_rag",
        pdfs=pdfs,
        qa_path=str(qa_path),
        chunk_size=300,
        chunk_overlap=50,
        out_dir=out_dir,
        mode="chunk_vectors",
        top_k=3,
    )
    write_report(res["results"], out_dir)

    memos_url = os.getenv("MEMOS_URL")
    if memos_url:
        try:
            resp = requests.get(memos_url, timeout=3)
            if resp.status_code < 500:
                run_benchmark(
                    backend_name="memos_rag",
                    pdfs=pdfs,
                    qa_path=str(qa_path),
                    chunk_size=300,
                    chunk_overlap=50,
                    out_dir=out_dir,
                    mode="chunk_vectors",
                    top_k=3,
                    memos_url=memos_url,
                    memos_user_id=os.getenv("MEMOS_USER_ID", "default"),
                    memos_cube_id=os.getenv("MEMOS_CUBE_ID", "default"),
                    memos_async_mode=os.getenv("MEMOS_ASYNC_MODE", "sync"),
                )
        except Exception:
            pass

    assert (Path(out_dir) / "report.md").exists()
    assert (Path(out_dir) / "summary.json").exists()
    summary = json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))
    backend = summary.get("backends", {}).get("sqlite_rag", {})
    for key in ["avg_exact_match", "retrieval_recall_rate", "citation_hit_rate"]:
        assert key in backend
    print("Smoke test complete. Results written to out/")


if __name__ == "__main__":
    main()
