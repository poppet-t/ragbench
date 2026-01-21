from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorStore(ABC):
    @abstractmethod
    def index(self, items: List[Dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        ...
