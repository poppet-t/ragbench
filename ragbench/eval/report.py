import json
import os
from typing import List, Dict, Any, Optional


def _avg_metric(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [row["metrics"].get(key) for row in rows]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _avg_bool(rows: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        val = row["metrics"].get(key)
        if val is None:
            continue
        vals.append(1.0 if val else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def _collect_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latency_retrieval_p50 = percentile([r["latency_ms_retrieval"] for r in rows], 50)
    latency_retrieval_p95 = percentile([r["latency_ms_retrieval"] for r in rows], 95)
    latency_p50 = percentile([r["latency_ms_total"] for r in rows], 50)
    latency_p95 = percentile([r["latency_ms_total"] for r in rows], 95)
    return {
        "count": len(rows),
        "avg_exact_match": _avg_bool(rows, "exact_match"),
        "avg_semantic_similarity": _avg_metric(rows, "semantic_similarity"),
        "retrieval_recall_rate": _avg_bool(rows, "retrieval_recall_at_k"),
        "citation_hit_rate": _avg_bool(rows, "citation_hit"),
        "citation_precision": _avg_metric(rows, "citation_precision"),
        "citation_recall": _avg_metric(rows, "citation_recall"),
        "groundedness_pass_rate": _avg_bool(rows, "groundedness_pass"),
        "groundedness_score": _avg_metric(rows, "groundedness_score"),
        "numeric_exact_rate": _avg_bool(rows, "numeric_exact"),
        "numeric_f1": _avg_metric(rows, "numeric_f1"),
        "abstain_correct_rate": _avg_bool(rows, "abstain_correct"),
        "false_abstain_rate": _avg_bool(rows, "false_abstain"),
        "latency_retrieval_p50": latency_retrieval_p50,
        "latency_retrieval_p95": latency_retrieval_p95,
        "latency_p50": latency_p50,
        "latency_p95": latency_p95,
        "query_p50_ms": latency_retrieval_p50,
        "query_p95_ms": latency_retrieval_p95,
        "citation_page_distance_avg": _avg_citation_distance(rows),
    }


def _avg_citation_distance(rows: List[Dict[str, Any]]) -> float:
    vals = [row["metrics"].get("citation_page_distance") for row in rows]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_backend: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_backend.setdefault(r["backend"], []).append(r)
    backends: Dict[str, Any] = {}
    for backend, rows in by_backend.items():
        backends[backend] = _collect_summary(rows)
        pipeline = rows[0].get("pipeline", {}) if rows else {}
        backends[backend]["index_build_ms"] = pipeline.get("index_build_ms")
        backends[backend]["index_size_bytes"] = pipeline.get("index_size_bytes")
        backends[backend]["embed_calls"] = pipeline.get("embed_calls")
        backends[backend]["llm_index_calls"] = pipeline.get("llm_index_calls")
        backends[backend]["memos_index_time_ms"] = pipeline.get("memos_index_time_ms")
        backends[backend]["memos_add_calls"] = pipeline.get("memos_add_calls")
        backends[backend]["memos_search_calls"] = pipeline.get("memos_search_calls")

    return {
        "backends": backends,
        "by_doc_id": _group_summary(results, "doc_id"),
        "by_backend_doc_id": _group_summary_backend(results, "doc_id"),
        "by_hop": _group_summary(results, "hop"),
        "by_negative": _group_summary(results, "is_negative"),
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[int(k)])
    return float(values[f] * (c - k) + values[c] * (k - f))


def write_report(results: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary = aggregate(results)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    lines = ["# RAG Benchmark Report", ""]
    lines.append("| Backend | N | Exact match | Sem sim | Recall@k | Citation hit | Grounded | p50 ms | p95 ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for backend, s in summary["backends"].items():
        lines.append(
            f"| {backend} | {s['count']} | {s['avg_exact_match']:.2f} | {s['avg_semantic_similarity']:.2f} | "
            f"{s['retrieval_recall_rate']:.2f} | {s['citation_hit_rate']:.2f} | {s['groundedness_pass_rate']:.2f} | "
            f"{s['latency_p50']:.1f} | {s['latency_p95']:.1f} |"
        )
    lines.append("")
    lines.append("## Index stats")
    lines.append("| Backend | Build ms | Index size (bytes) | Embed calls | LLM index calls |")
    lines.append("| --- | --- | --- | --- | --- |")
    for backend, s in summary["backends"].items():
        lines.append(
            f"| {backend} | {s.get('index_build_ms')} | {s.get('index_size_bytes')} | {s.get('embed_calls')} | {s.get('llm_index_calls')} |"
        )
    lines.append("")
    lines.append("## MemOS stats")
    lines.append("| Backend | MemOS index ms | MemOS add calls | MemOS search calls | Query p50 | Query p95 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for backend, s in summary["backends"].items():
        lines.append(
            f"| {backend} | {s.get('memos_index_time_ms')} | {s.get('memos_add_calls')} | {s.get('memos_search_calls')} | "
            f"{s.get('query_p50_ms'):.1f} | {s.get('query_p95_ms'):.1f} |"
        )
    lines.append("")
    lines.append("## Latency (retrieval vs total)")
    lines.append("| Backend | Retrieval p50 | Retrieval p95 | Total p50 | Total p95 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for backend, s in summary["backends"].items():
        lines.append(
            f"| {backend} | {s['latency_retrieval_p50']:.1f} | {s['latency_retrieval_p95']:.1f} | "
            f"{s['latency_p50']:.1f} | {s['latency_p95']:.1f} |"
        )
    lines.append("")
    lines.extend(_breakdown_section(results, "doc_id"))
    lines.append("")
    lines.extend(_breakdown_section(results, "hop"))
    lines.append("")
    lines.extend(_breakdown_section(results, "is_negative"))
    lines.append("")
    lines.extend(_numeric_section(results))
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _breakdown_section(results: List[Dict[str, Any]], key: str) -> List[str]:
    lines = [f"## Breakdown by {key}"]
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for r in results:
        group = str(r.get(key))
        grouped.setdefault((r.get("backend"), group), []).append(r)
    lines.append("| Backend | Group | N | Recall@k | Citation hit | Grounded | p50 ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for (backend, group), rows in grouped.items():
        s = _collect_summary(rows)
        lines.append(
            f"| {backend} | {group} | {s['count']} | {s['retrieval_recall_rate']:.2f} | {s['citation_hit_rate']:.2f} | "
            f"{s['groundedness_pass_rate']:.2f} | {s['latency_p50']:.1f} |"
        )
    return lines


def _numeric_section(results: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Numeric fidelity (earnings)"]
    rows = [r for r in results if r.get("doc_id") == "earnings"]
    if not rows:
        lines.append("No earnings questions found.")
        return lines
    lines.append("| Backend | N | Numeric exact | Numeric F1 |")
    lines.append("| --- | --- | --- | --- |")
    by_backend: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)
    for backend, items in by_backend.items():
        f1 = _avg_metric(items, "numeric_f1")
        exact = _avg_bool(items, "numeric_exact")
        lines.append(f"| {backend} | {len(items)} | {exact:.2f} | {f1:.2f} |")
    return lines


def _group_summary(results: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(str(r.get(key)), []).append(r)
    return {k: _collect_summary(v) for k, v in grouped.items()}


def _group_summary_backend(results: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in results:
        backend = r.get("backend")
        group = str(r.get(key))
        grouped.setdefault(backend, {}).setdefault(group, []).append(r)
    out: Dict[str, Any] = {}
    for backend, groups in grouped.items():
        out[backend] = {g: _collect_summary(rows) for g, rows in groups.items()}
    return out
