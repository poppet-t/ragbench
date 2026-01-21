import math
import re
from typing import List, Dict, Any, Tuple, Optional

from ..embeddings import hash_embedding


_STOPWORDS = {
    "the", "and", "or", "for", "to", "of", "in", "on", "by", "a", "an", "is", "are", "was", "were",
    "with", "as", "that", "this", "it", "be", "from", "at", "their", "they", "we", "what", "which",
    "who", "how", "does", "did", "when", "where", "why", "about",
}


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def exact_match(pred: str, truth: str) -> bool:
    return normalize(pred) == normalize(truth)


def semantic_similarity(pred: str, truth: str) -> float:
    vecs = hash_embedding([pred, truth])
    a, b = vecs
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def parse_citation_range(cite: str) -> Tuple[str, Optional[int], Optional[int]]:
    m = re.search(r"(.+?)\s+p(\d+)(?:-(\d+))?", cite)
    if not m:
        return cite.strip(), None, None
    pdf = m.group(1).strip()
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    return pdf, start, end


def _expand_citations(cites: List[str]) -> List[Tuple[str, int]]:
    expanded: List[Tuple[str, int]] = []
    for cite in cites:
        pdf, start, end = parse_citation_range(cite)
        if start is None or end is None:
            continue
        span = range(start, end + 1)
        if end - start > 100:
            span = range(start, start + 1)
        for p in span:
            expanded.append((pdf, p))
    return expanded


def citation_hit(pred_cites: List[str], gt_cites: List[str]) -> bool:
    pred = set(_expand_citations(pred_cites))
    gt = set(_expand_citations(gt_cites))
    return bool(pred.intersection(gt))


def citation_precision(pred_cites: List[str], gt_cites: List[str]) -> float:
    pred = set(_expand_citations(pred_cites))
    gt = set(_expand_citations(gt_cites))
    if not pred:
        return 0.0
    return len(pred.intersection(gt)) / len(pred)


def citation_recall(pred_cites: List[str], gt_cites: List[str]) -> float:
    pred = set(_expand_citations(pred_cites))
    gt = set(_expand_citations(gt_cites))
    if not gt:
        return 0.0
    return len(pred.intersection(gt)) / len(gt)


def citation_page_distance(pred_cites: List[str], gt_cites: List[str]) -> Optional[int]:
    pred = _expand_citations(pred_cites)
    gt = _expand_citations(gt_cites)
    if not pred or not gt:
        return None
    distances = []
    for pdf_p, page_p in pred:
        for pdf_g, page_g in gt:
            if pdf_p != pdf_g:
                continue
            distances.append(abs(page_p - page_g))
    return min(distances) if distances else None


def retrieval_recall_at_k(retrieved: List[Dict[str, Any]], gt_cites: List[str]) -> bool:
    gt = _expand_citations(gt_cites)
    for hit in retrieved:
        for pdf, page in gt:
            if pdf and hit.get("pdf_filename") and pdf.strip() != hit.get("pdf_filename").strip():
                continue
            if hit.get("page_start") is not None and hit.get("page_end") is not None:
                if hit["page_start"] <= page <= hit["page_end"]:
                    return True
    return False


def _content_words(text: str) -> List[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _extract_numbers(text: str) -> List[str]:
    raw = re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", text)
    cleaned = []
    for n in raw:
        n = n.replace(",", "").replace("$", "")
        if n.endswith("%"):
            n = n[:-1]
        cleaned.append(n)
    return cleaned


def numeric_fidelity(pred: str, truth: str) -> Tuple[Optional[bool], Optional[float]]:
    gt_nums = _extract_numbers(truth)
    if not gt_nums:
        return None, None
    pred_nums = _extract_numbers(pred)
    gt_set = set(gt_nums)
    pred_set = set(pred_nums)
    exact = gt_set.issubset(pred_set) and bool(pred_set)
    if not pred_set:
        return False, 0.0
    match = len(gt_set.intersection(pred_set))
    precision = match / len(pred_set)
    recall = match / len(gt_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return exact, f1


def groundedness(qa_record: Dict[str, Any], answer: str, retrieved: List[Dict[str, Any]]) -> Tuple[bool, float]:
    if "not found" in answer.lower():
        return qa_record.get("is_negative", False), 1.0 if qa_record.get("is_negative", False) else 0.0
    evidence = " ".join(r.get("text", "") for r in retrieved).lower()
    ans_words = _content_words(answer)
    if not ans_words:
        return False, 0.0
    overlap = sum(1 for w in ans_words if w in evidence)
    ratio = overlap / max(1, len(ans_words))
    ans_nums = _extract_numbers(answer)
    for n in ans_nums:
        if n and n not in evidence:
            return False, 0.0
    return ratio >= 0.2, ratio


def abstain_metrics(qa_record: Dict[str, Any], answer: str) -> Tuple[bool, bool]:
    abstain = "not found in the provided documents" in answer.lower()
    is_negative = qa_record.get("is_negative", False)
    abstain_correct = bool(is_negative and abstain)
    false_abstain = bool((not is_negative) and abstain)
    return abstain_correct, false_abstain
