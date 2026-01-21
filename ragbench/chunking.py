import re
from typing import List, Dict, Any


def _approx_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _looks_like_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    if re.match(r"^[0-9IVXLC]+\.", stripped):
        return True
    return stripped.isupper()


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 120, method: str = "default") -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        paragraphs = _split_paragraphs(page["text"])
        if method == "by_section_headers":
            temp: List[str] = []
            for para in paragraphs:
                lines = para.splitlines()
                if lines and _looks_like_header(lines[0]) and temp:
                    chunks.extend(
                        _chunk_from_paras(temp, page, chunk_size, chunk_overlap)
                    )
                    temp = []
                temp.append(para)
            if temp:
                chunks.extend(_chunk_from_paras(temp, page, chunk_size, chunk_overlap))
        else:
            chunks.extend(_chunk_from_paras(paragraphs, page, chunk_size, chunk_overlap))
    return chunks


def _chunk_from_paras(paragraphs: List[str], page: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    para_tokens = [(p, _approx_tokens(p)) for p in paragraphs]
    idx = 0
    out: List[Dict[str, Any]] = []
    chunk_idx = 0
    while idx < len(para_tokens):
        toks = 0
        buff: List[str] = []
        j = idx
        while j < len(para_tokens):
            p, t = para_tokens[j]
            if toks > 0 and toks + t > chunk_size:
                break
            buff.append(p)
            toks += t
            j += 1
        if not buff:
            buff.append(para_tokens[j][0])
            j += 1
        chunk_idx += 1
        text = "\n\n".join(buff)
        out.append(
            {
                "chunk_id": f"{page['doc_id']}_p{page['page_num']}_c{chunk_idx}",
                "doc_id": page["doc_id"],
                "pdf_filename": page["pdf_filename"],
                "page_start": page["page_num"],
                "page_end": page["page_num"],
                "chunk_index": chunk_idx,
                "text": text,
            }
        )
        overlap_tokens = 0
        overlap_paras = 0
        for para_text, t in reversed(para_tokens[idx:j]):
            overlap_tokens += t
            if overlap_tokens > chunk_overlap:
                break
            overlap_paras += 1
        overlap_paras = min(overlap_paras, max(0, (j - idx) - 1))
        idx = j - overlap_paras
    return out
