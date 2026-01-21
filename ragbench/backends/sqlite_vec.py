import json
import math
import sqlite3
from typing import List, Dict, Any

from .base import VectorStore


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class SQLiteVectorStore(VectorStore):
    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items(
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                pdf_filename TEXT,
                page_start INT,
                page_end INT,
                text TEXT,
                meta TEXT,
                embedding TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def index(self, items: List[Dict[str, Any]]) -> None:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        for item in items:
            cur.execute(
                """
                INSERT OR REPLACE INTO items(chunk_id, doc_id, pdf_filename, page_start, page_end, text, meta, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["chunk_id"],
                    item["doc_id"],
                    item.get("pdf_filename"),
                    item.get("page_start"),
                    item.get("page_end"),
                    item.get("text", ""),
                    json.dumps(item.get("meta") or {}),
                    json.dumps(item["embedding"]),
                ),
            )
        conn.commit()
        conn.close()

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, doc_id, pdf_filename, page_start, page_end, text, meta, embedding FROM items")
        rows = cur.fetchall()
        conn.close()
        scored: List[Dict[str, Any]] = []
        for chunk_id, doc_id, pdf_filename, p_s, p_e, text, meta_json, emb_json in rows:
            emb = json.loads(emb_json)
            score = _cosine(query_vector, emb)
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "pdf_filename": pdf_filename,
                    "page_start": p_s,
                    "page_end": p_e,
                    "text": text,
                    "meta": json.loads(meta_json or "{}"),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
