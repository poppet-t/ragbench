# RAGBench (SQLite vs Milvus vs PageIndex-style vs MemOS)

Offline-friendly benchmark comparing multiple retrieval/indexing approaches (including MemOS) on three PDFs.
Run commands from the repo root. If you see `No module named scripts`, prefix with `PYTHONPATH=.`.

## Quickstart (mock LLM + hash embeddings)
```
python -m scripts.run_eval \
  --backend sqlite_rag milvus_rag pageindex_sqlite \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --out-dir out
```
Outputs in `out/` (`*_results.json`, `summary.json`, `report.md`).

## Reduced vs full QA set
- Full: `data/qa_seed.jsonl` (default)
- Reduced 25: `data/qa_reduced_25.jsonl`

Use `--qa-set` to switch:
```
python -m scripts.run_eval \
  --backend sqlite_rag milvus_rag pageindex_sqlite memos_rag \
  --qa-set reduced \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --out-dir out
```

Reduced 25 IDs (balanced across 3 PDFs + mixed negatives):
```
wp001 wp002 wp006 wp008 wp017 wp023 wp029
er001 er002 er004 er005 er014 er021 er028
sp001 sp004 sp005 sp006 sp010 sp017 sp020
mx001 mx004 mx007 mx009
```

## Quality run (real embeddings + LLM answers)
Recommended for improved accuracy while keeping PageIndex cards cheap:
```
python -m scripts.run_eval \
  --backend sqlite_rag milvus_rag pageindex_sqlite \
  --qa-set reduced \
  --embedding-provider sentence_transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --llm-index-provider mock \
  --llm-answer-provider openai \
  --llm-answer-model gpt-4o-mini \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --out-dir out
```
Results are written to `out/report.md` and `out/summary.json`.

To use a real local embedding model:
```
python -m scripts.run_eval \
  --backend sqlite_rag \
  --embedding-provider sentence_transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --out-dir out
```

## Using local Qwen 4B (Transformers)
Make sure the model is available in your HF cache or provide a local path. This will be used for answers and PageIndex index cards.
```
python -m scripts.run_eval \
  --backend sqlite_rag pageindex_sqlite \
  --llm-provider qwen \
  --llm-model Qwen/Qwen2.5-4B-Instruct-2507 \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --out-dir out
```
Optional: set `LLM_DEVICE=mps|cuda|cpu` to control where the model runs.

## Using Ollama (local HTTP)
If you installed Qwen via Ollama, use the Ollama adapter:
```
python -m scripts.run_eval \
  --backend sqlite_rag pageindex_sqlite \
  --llm-provider ollama \
  --llm-model qwen2.5:4b \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --out-dir out
```
Set `OLLAMA_URL` if your server is not `http://localhost:11434`.

## Using OpenAI (optional)
Install the client and set your API key:
```
pip install openai
export OPENAI_API_KEY=...   # do not commit keys
```
Then run:
```
python -m scripts.run_eval \
  --backend sqlite_rag pageindex_sqlite \
  --llm-index-provider mock \
  --llm-answer-provider openai \
  --llm-answer-model gpt-4o-mini \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --out-dir out
```
If you want OpenAI for both index cards and answers, use `--llm-provider openai --llm-model gpt-4o-mini`.

## Separate LLMs for index cards vs answers
You can keep PageIndex index cards cheap while using a stronger model for answers:
```
python -m scripts.run_eval \
  --backend pageindex_sqlite \
  --llm-index-provider mock \
  --llm-answer-provider ollama \
  --llm-answer-model qwen3:4b-thinking-2507-q4_K_M \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --out-dir out
```

## Index caching
If the PDFs + chunking + embedding config haven't changed, the index will be reused automatically.
Use `--force-reindex` to rebuild from scratch.
For PageIndex, index cards are cached per LLM provider/model under `out/<backend>_cards_<provider>_<model>.jsonl`.

## What’s included
- Ingestion: PyMuPDF page extraction.
- Chunking: page-aware, target size/overlap.
- Embeddings: deterministic hash baseline (default) or pluggable HF/sentence-transformers.
- LLM: mock (extractive), local Qwen (Transformers), or Ollama.
- Backends: SQLite vector, Milvus, PageIndex-style (index cards + retrieval), MemOS HTTP.
- Metrics: recall@k, citation hit, exact match, semantic similarity, groundedness, numeric fidelity, latencies.

## Building indexes only
```
python -m scripts.build_index --backend sqlite --pdfs ... --doc-ids ... --chunk-size 800 --chunk-overlap 120
```
Backends: `sqlite_rag|milvus_rag|pageindex_sqlite|pageindex_milvus`.

## Running evaluation
```
python -m scripts.run_eval \
  --backend pageindex_sqlite \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --k 5 \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --abstain-threshold 0.1 \
  --out-dir out
```

## Milvus setup
- Start with Docker Compose: `docker compose -f docker/docker-compose.milvus.yml up -d`
- Use `--backend milvus_rag` or `--backend pageindex_milvus`.

## MemOS backend
If you have a MemOS HTTP service running:
```
python -m scripts.run_eval \
  --backend memos_rag \
  --memos-url http://localhost:8000 \
  --memos-user-id default \
  --memos-cube-id default \
  --memos-async-mode sync \
  --pdfs /mnt/data/zCloak_AI_Whitepaper_v1.4.pdf /mnt/data/zhang_uafx\\ \\(1\\).pdf /mnt/data/NVIDIAAn.pdf \
  --doc-ids whitepaper paper earnings \
  --qa data/qa_seed.jsonl \
  --out-dir out
```
Tips:
- Use a fresh `--memos-user-id` and `--memos-cube-id` per run to avoid mixing old memory formats.
- When switching configs or after code changes, add `--force-reindex`.
- Sanity check the service first: `curl http://localhost:8000/docs`.

## Data/QA
- Seed QA: `data/qa_seed.jsonl` (provided).
- Add more questions by appending lines with `id, doc_id, question, ground_truth_answer, ground_truth_citations`.

## Design notes
- PageIndex-style: generates index cards via the configured LLM, embeds `embedding_text`, retrieves index cards, then fetches original chunks for grounding.
- Mock LLM + hash embeddings ensure offline reproducibility; swap providers via CLI flags for higher quality runs.
- Reports: `out/results.json` (all results), `out/summary.json`, `out/report.md`.

## Metrics (high level)
- Retrieval: recall@k, citation precision/recall, citation page distance.
- Answer quality: exact match, semantic similarity, groundedness score, numeric fidelity.
- Robustness: abstain correctness for negative questions, false abstain rate.
- Latency: retrieval + end-to-end p50/p95.

## Scripts
- `scripts/build_index.py`: build chosen backend index.
- `scripts/run_eval.py`: end-to-end benchmark.
- `scripts/smoke_test.py`: minimal run on the whitepaper only.
