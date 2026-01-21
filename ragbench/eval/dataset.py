import json
from typing import List, Dict, Any


def _is_negative(row: Dict[str, Any]) -> bool:
    if not row.get("ground_truth_citations"):
        return True
    question = (row.get("question") or "").lower()
    answer = (row.get("ground_truth_answer") or "").lower()
    if any(k in question for k in ["not found", "does it specify", "does it mention"]) and "not found" in answer:
        return True
    return False


def _hop_level(question: str) -> int:
    q = question.lower()
    if q.startswith("multi-hop"):
        return 2
    if any(k in q for k in ["compare", "relative to", "across", "both", "difference between"]):
        return 2
    return 1


def load_qa(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["is_negative"] = _is_negative(row)
            row["hop"] = _hop_level(row.get("question", ""))
            rows.append(row)
    return rows
