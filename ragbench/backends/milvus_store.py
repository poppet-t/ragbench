from typing import List, Dict, Any

from .base import VectorStore


class MilvusVectorStore(VectorStore):
    def __init__(self, uri: str = "http://localhost:19530", collection: str = "ragbench", dim: int = 128):
        self.uri = uri
        self.collection_name = collection
        self.dim = dim
        self._milvus = None
        self._try_connect()
        self._memory: List[Dict[str, Any]] = []

    def _try_connect(self):
        try:
            from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility  # type: ignore

            connections.connect(alias="default", uri=self.uri)
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="pdf_filename", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="page_start", dtype=DataType.INT64),
                FieldSchema(name="page_end", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(fields)
            if utility.has_collection(self.collection_name):
                self._milvus = Collection(self.collection_name)
            else:
                self._milvus = Collection(self.collection_name, schema)
                self._milvus.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}})
            self._milvus.load()
        except Exception:
            self._milvus = None

    def index(self, items: List[Dict[str, Any]]) -> None:
        if self._milvus is None:
            self._memory.extend(items)
            return
        max_text_len = 8192

        def _truncate_utf8(text: str, max_bytes: int) -> str:
            raw = text.encode("utf-8")
            if len(raw) <= max_bytes:
                return text
            return raw[:max_bytes].decode("utf-8", errors="ignore")

        data = [
            [it["chunk_id"] for it in items],
            [it["doc_id"] for it in items],
            [it.get("pdf_filename") for it in items],
            [it.get("page_start") for it in items],
            [it.get("page_end") for it in items],
            [_truncate_utf8((it.get("text", "") or ""), max_text_len) for it in items],
            [it["embedding"] for it in items],
        ]
        self._milvus.insert(data)  # type: ignore
        self._milvus.flush()  # type: ignore

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        if self._milvus is None:
            # fallback linear search
            from .sqlite_vec import _cosine

            scored = []
            for it in self._memory:
                score = _cosine(query_vector, it["embedding"])
                out = it.copy()
                out["score"] = score
                scored.append(out)
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self._milvus.search(  # type: ignore
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "pdf_filename", "page_start", "page_end", "text"],
        )[0]
        out: List[Dict[str, Any]] = []
        for hit in results:
            out.append(
                {
                    "chunk_id": hit.entity.get("chunk_id"),
                    "doc_id": hit.entity.get("doc_id"),
                    "pdf_filename": hit.entity.get("pdf_filename"),
                    "page_start": hit.entity.get("page_start"),
                    "page_end": hit.entity.get("page_end"),
                    "text": hit.entity.get("text"),
                    "score": float(hit.score),
                }
            )
        return out
