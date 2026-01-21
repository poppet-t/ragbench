import hashlib
import math
from typing import List, Iterable


def hash_embedding(texts: Iterable[str], dim: int = 128) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        vec = [0.0] * dim
        h = hashlib.sha1(text.encode("utf-8")).digest()
        for i, b in enumerate(h):
            vec[i % dim] += (b - 127.5) / 255.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


class Embedder:
    def __init__(self, provider: str = "hash", model: str | None = None):
        self.provider = provider
        self.model = model
        self._backend = None
        self.embed_calls = 0
        self.embed_texts = 0

    def _load_backend(self):
        if self._backend is not None:
            return
        if self.provider == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise ImportError("sentence-transformers not installed") from exc
            self._backend = SentenceTransformer(self.model or "sentence-transformers/all-MiniLM-L6-v2")
        elif self.provider == "huggingface":
            try:
                import torch  # type: ignore
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise ImportError("transformers+torch not installed") from exc
            tok = AutoTokenizer.from_pretrained(self.model)
            mod = AutoModel.from_pretrained(self.model)
            self._backend = (tok, mod)
        else:
            self._backend = "hash"

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.embed_calls += 1
        self.embed_texts += len(texts)
        self._load_backend()
        if self._backend == "hash":
            return hash_embedding(texts)
        if isinstance(self._backend, tuple):
            tok, mod = self._backend
            import torch  # type: ignore

            enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = mod(**enc).last_hidden_state.mean(dim=1)
            return out.cpu().tolist()
        return self._backend.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
