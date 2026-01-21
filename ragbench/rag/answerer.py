from typing import List, Dict, Any
import re
import time


_STOPWORDS = {
    "the", "and", "or", "for", "to", "of", "in", "on", "by", "a", "an", "is", "are", "was", "were",
    "with", "as", "that", "this", "it", "be", "from", "at", "their", "they", "we", "what", "which",
    "who", "how", "does", "did", "when", "where", "why", "about",
}


def _content_words(text: str) -> List[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _is_numeric_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["how many", "percent", "%", "revenue", "eps", "income", "million", "billion", "ratio", "rate"])


def _is_definition_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["what is", "what are", "define", "means", "defined as"])


def _sentence_score(sentence: str, q_words: List[str], numeric: bool, definition: bool) -> float:
    s_words = set(_content_words(sentence))
    overlap = len(s_words.intersection(q_words))
    score = float(overlap)
    if numeric and re.search(r"\d", sentence):
        score += 1.0
    if definition and re.search(r"\b(is|are|means|defined as)\b", sentence.lower()):
        score += 0.5
    return score


def answer_question(
    question: str,
    contexts: List[Dict[str, Any]],
    abstain_threshold: float = 0.1,
    llm=None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    start = time.perf_counter()
    if not contexts:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "final_answer": "Not found in the provided documents.",
            "citations": [],
            "rationale": "No retrieved context.",
            "latency_ms_answer": latency_ms,
        }

    q_words = _content_words(question)
    numeric = _is_numeric_question(question)
    definition = _is_definition_question(question)

    best = {"score": -1.0, "sentence": "", "citation": "", "chunk": None}
    top_retrieval_score = max((c.get("score", 0.0) for c in contexts), default=0.0)
    for ctx in contexts:
        sentences = re.split(r"(?<=[.!?])\s+", ctx.get("text", ""))
        for sent in sentences:
            score = _sentence_score(sent, q_words, numeric, definition)
            if score > best["score"]:
                best = {
                    "score": score,
                    "sentence": sent.strip(),
                    "citation": f"{ctx.get('pdf_filename')} p{ctx.get('page_start')}",
                    "chunk": ctx,
                }

    overlap_ratio = best["score"] / max(1, len(q_words)) if q_words else 0.0
    if top_retrieval_score < 0.1 or overlap_ratio < abstain_threshold or not best["sentence"]:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "final_answer": "Not found in the provided documents.",
            "citations": [],
            "rationale": "Low evidence overlap with question.",
            "latency_ms_answer": latency_ms,
        }

    if use_llm and llm is not None:
        try:
            llm_start = time.perf_counter()
            resp = llm.answer(question, contexts)
            resp["latency_ms_answer"] = (time.perf_counter() - llm_start) * 1000
            return resp
        except Exception:
            pass

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "final_answer": best["sentence"],
        "citations": [best["citation"]],
        "rationale": "Selected the highest-overlap sentence from retrieved chunks.",
        "latency_ms_answer": latency_ms,
    }
