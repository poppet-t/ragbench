import json
import os
import re
from typing import List, Dict, Any

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MockLLM:
    """Deterministic mock LLM for offline use."""

    def __init__(self, model: str | None = None):
        self.model = model or "mock-llm"
        self.index_calls = 0
        self.answer_calls = 0

    def _extract_keywords(self, text: str, max_keywords: int = 6) -> List[str]:
        stop = {
            "the", "and", "or", "for", "to", "of", "in", "on", "by", "a", "an", "is", "are", "was",
            "were", "with", "as", "that", "this", "it", "be", "from", "at", "their", "they", "we",
        }
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]
        freq: Dict[str, int] = {}
        for w in words:
            if len(w) < 3 or w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [w for w, _ in ranked[:max_keywords]]

    def generate_index_card(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        self.index_calls += 1
        text = chunk["text"]
        first_sentence = text.strip().split(".")[0][:200]
        keywords = self._extract_keywords(text)
        card_text = "Summary: " + (first_sentence.strip() or "No summary.") + "\n"
        card_text += "Keywords: " + ", ".join(keywords) + "\n"
        card_text += "Suggested Questions: " + "; ".join([f"What about {k}?" for k in keywords[:3]])
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "title_guess": first_sentence.strip() or "Section",
            "key_facts": [first_sentence.strip()] if first_sentence else [],
            "entities": keywords,
            "suggested_questions": [f"What about {k}?" for k in keywords[:3]],
            "embedding_text": card_text.strip(),
        }

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.answer_calls += 1
        joined = " ".join(c["text"] for c in contexts)[:500]
        citation_strings = []
        for c in contexts:
            citation_strings.append(f"{c['pdf_filename']} p{c['page_start']}")
        final = joined or "Not found in documents."
        return {
            "final_answer": final,
            "citations": citation_strings[:3],
            "rationale": "Extracted from retrieved chunks.",
        }


class LocalQwenLLM:
    """Local Qwen model adapter via transformers."""

    def __init__(self, model: str | None = None, device: str | None = None, max_new_tokens: int = 256):
        self.model_name = model or "Qwen/Qwen2.5-4B-Instruct-2507"
        self.device = device or self._default_device()
        self.max_new_tokens = max_new_tokens
        self.index_calls = 0
        self.answer_calls = 0
        self._tokenizer = None
        self._model = None

    def _default_device(self) -> str:
        env_device = os.getenv("LLM_DEVICE")
        if env_device:
            return env_device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        return {}

    def _ensure_str_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if value is None:
            return []
        return [str(value).strip()]

    def _format_chat(self, system: str, user: str) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return system + "\n\n" + user + "\n\n"

    def _generate(self, system: str, user: str) -> str:
        self._load()
        prompt = self._format_chat(system, user)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_index_card(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        self.index_calls += 1
        text = chunk.get("text", "")
        excerpt = text[:2000]
        system = (
            "You create compact index cards for retrieval. Return only JSON with keys: "
            "title_guess (string), key_facts (list of strings), entities (list of strings), "
            "suggested_questions (list of strings), embedding_text (string). "
            "Keep lists short (max 5). embedding_text should be keyword-rich."
        )
        user = (
            f"doc_id: {chunk.get('doc_id')}\n"
            f"pages: {chunk.get('page_start')} - {chunk.get('page_end')}\n"
            f"chunk_text:\n{excerpt}"
        )
        content = self._generate(system, user)
        data = self._parse_json(content)
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "title_guess": data.get("title_guess") or "Index Card",
            "key_facts": self._ensure_str_list(data.get("key_facts")),
            "entities": self._ensure_str_list(data.get("entities")),
            "suggested_questions": self._ensure_str_list(data.get("suggested_questions")),
            "embedding_text": data.get("embedding_text") or excerpt[:300],
        }

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.answer_calls += 1
        ctx_lines = []
        for idx, ctx in enumerate(contexts, start=1):
            citation = f"{ctx.get('pdf_filename')} p{ctx.get('page_start')}"
            snippet = (ctx.get("text") or "")[:1200]
            ctx_lines.append(f"[{idx}] {citation}\n{snippet}")
        system = (
            "You are a customer support assistant. Use only the provided contexts. "
            "Return JSON with keys: final_answer (string), citations (list of strings), rationale (string). "
            "If the answer is not in the contexts, respond with final_answer "
            "'Not found in the provided documents.' and empty citations."
        )
        user = f"Question: {question}\n\nContexts:\n" + "\n\n".join(ctx_lines)
        content = self._generate(system, user)
        data = self._parse_json(content)
        citations = self._ensure_str_list(data.get("citations"))
        return {
            "final_answer": data.get("final_answer") or "Not found in the provided documents.",
            "citations": citations,
            "rationale": data.get("rationale") or "Answered using provided contexts.",
        }


class OllamaLLM:
    """Ollama adapter (local HTTP)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout_s: int = 120):
        self.model = model or "qwen2.5:4b"
        self.base_url = (base_url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.timeout_s = timeout_s
        self.index_calls = 0
        self.answer_calls = 0

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        return {}

    def _ensure_str_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if value is None:
            return []
        return [str(value).strip()]

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.0, "top_p": 1.0},
        }
        data = self._post("/api/chat", payload)
        msg = data.get("message") or {}
        return msg.get("content", "")

    def generate_index_card(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        self.index_calls += 1
        text = chunk.get("text", "")
        excerpt = text[:2000]
        system = (
            "You create compact index cards for retrieval. Return only JSON with keys: "
            "title_guess (string), key_facts (list of strings), entities (list of strings), "
            "suggested_questions (list of strings), embedding_text (string). "
            "Keep lists short (max 5). embedding_text should be keyword-rich."
        )
        user = (
            f"doc_id: {chunk.get('doc_id')}\n"
            f"pages: {chunk.get('page_start')} - {chunk.get('page_end')}\n"
            f"chunk_text:\n{excerpt}"
        )
        content = self._chat(system, user)
        data = self._parse_json(content)
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "title_guess": data.get("title_guess") or "Index Card",
            "key_facts": self._ensure_str_list(data.get("key_facts")),
            "entities": self._ensure_str_list(data.get("entities")),
            "suggested_questions": self._ensure_str_list(data.get("suggested_questions")),
            "embedding_text": data.get("embedding_text") or excerpt[:300],
        }

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.answer_calls += 1
        ctx_lines = []
        for idx, ctx in enumerate(contexts, start=1):
            citation = f"{ctx.get('pdf_filename')} p{ctx.get('page_start')}"
            snippet = (ctx.get("text") or "")[:1200]
            ctx_lines.append(f"[{idx}] {citation}\n{snippet}")
        system = (
            "You are a customer support assistant. Use only the provided contexts. "
            "Return JSON with keys: final_answer (string), citations (list of strings), rationale (string). "
            "If the answer is not in the contexts, respond with final_answer "
            "'Not found in the provided documents.' and empty citations."
        )
        user = f"Question: {question}\n\nContexts:\n" + "\n\n".join(ctx_lines)
        content = self._chat(system, user)
        data = self._parse_json(content)
        citations = self._ensure_str_list(data.get("citations"))
        return {
            "final_answer": data.get("final_answer") or "Not found in the provided documents.",
            "citations": citations,
            "rationale": data.get("rationale") or "Answered using provided contexts.",
        }


class OpenAILLM:
    """OpenAI adapter (optional, requires openai package)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: int = 120,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("openai package not installed. Run: pip install openai") from exc
        self.model = model or "gpt-4o-mini"
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.timeout_s = timeout_s
        self.index_calls = 0
        self.answer_calls = 0

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        return {}

    def _ensure_str_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if value is None:
            return []
        return [str(value).strip()]

    def _chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            top_p=1.0,
        )
        msg = resp.choices[0].message
        return (msg.content or "").strip()

    def generate_index_card(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        self.index_calls += 1
        text = chunk.get("text", "")
        excerpt = text[:2000]
        system = (
            "You create compact index cards for retrieval. Return only JSON with keys: "
            "title_guess (string), key_facts (list of strings), entities (list of strings), "
            "suggested_questions (list of strings), embedding_text (string). "
            "Keep lists short (max 5). embedding_text should be keyword-rich."
        )
        user = (
            f"doc_id: {chunk.get('doc_id')}\n"
            f"pages: {chunk.get('page_start')} - {chunk.get('page_end')}\n"
            f"chunk_text:\n{excerpt}"
        )
        content = self._chat(system, user)
        data = self._parse_json(content)
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "title_guess": data.get("title_guess") or "Index Card",
            "key_facts": self._ensure_str_list(data.get("key_facts")),
            "entities": self._ensure_str_list(data.get("entities")),
            "suggested_questions": self._ensure_str_list(data.get("suggested_questions")),
            "embedding_text": data.get("embedding_text") or excerpt[:300],
        }

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.answer_calls += 1
        ctx_lines = []
        for idx, ctx in enumerate(contexts, start=1):
            citation = f"{ctx.get('pdf_filename')} p{ctx.get('page_start')}"
            snippet = (ctx.get("text") or "")[:1200]
            ctx_lines.append(f"[{idx}] {citation}\n{snippet}")
        system = (
            "You are a customer support assistant. Use only the provided contexts. "
            "Return JSON with keys: final_answer (string), citations (list of strings), rationale (string). "
            "If the answer is not in the contexts, respond with final_answer "
            "'Not found in the provided documents.' and empty citations."
        )
        user = f"Question: {question}\n\nContexts:\n" + "\n\n".join(ctx_lines)
        content = self._chat(system, user)
        data = self._parse_json(content)
        citations = self._ensure_str_list(data.get("citations"))
        return {
            "final_answer": data.get("final_answer") or "Not found in the provided documents.",
            "citations": citations,
            "rationale": data.get("rationale") or "Answered using provided contexts.",
        }


class LLMFactory:
    def __init__(self, provider: str = "mock", model: str | None = None):
        self.provider = provider
        self.model = model

    def build(self):
        if self.provider == "mock":
            return MockLLM(model=self.model)
        if self.provider == "qwen":
            return LocalQwenLLM(model=self.model)
        if self.provider == "ollama":
            return OllamaLLM(model=self.model)
        if self.provider == "openai":
            return OpenAILLM(model=self.model)
        # Skeletons for real providers
        raise NotImplementedError(f"Unknown provider: {self.provider}")
