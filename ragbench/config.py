from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PDFSpec:
    doc_id: str
    path: str
    filename: str


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120
    method: str = "default"  # default|by_section_headers


@dataclass
class EmbeddingConfig:
    provider: str = "hash"  # hash|sentence_transformers|huggingface
    model: Optional[str] = None


@dataclass
class LLMConfig:
    provider: str = "mock"  # mock|qwen|ollama
    model: Optional[str] = None
    temperature: float = 0.0


@dataclass
class BenchmarkConfig:
    pdfs: List[PDFSpec] = field(default_factory=list)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    top_k: int = 5
