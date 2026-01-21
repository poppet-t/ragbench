from typing import List, Dict, Any


def extract_pages(pdf_path: str, doc_id: str, filename: str) -> List[Dict[str, Any]]:
    """Return list of pages with page_num (1-indexed) and text."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF extraction") from exc
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = (page.get_text("text") or "").replace("\u00ad", "")
        pages.append(
            {
                "doc_id": doc_id,
                "pdf_filename": filename,
                "page_num": i + 1,
                "text": text,
            }
        )
    return pages
