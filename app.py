# app.py
# ============================================================
# HỖ TRỢ NGHIÊN CỨU KHOA HỌC – EVIDENCE-BASED RAG
# Bản tối ưu cho luận văn Chuyên khoa cấp I – Dược lâm sàng
# ============================================================

import io
import os
import re
import time
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Google Gemini SDK mới: pip install google-genai
from google import genai
from google.genai import types

# Embedding & BM25
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# DOCX
from docx import Document

# ============================================================
# IMPORT TỪ CÁC MODULE ĐÃ ĐƯỢC BÓC TÁCH
# ============================================================
# 1. Import bộ máy tuyển chọn bảng
from table_selection_engine import (
    StudyObjective, CandidateResult,
    TableSelectionEngine, NarrativePlanner,
    Priority, Presentation
)

# 2. Import bộ máy Thống kê Toán học
from statistical_engine import (
    validate_dataframe, descriptive_table, numeric_summary,
    crosstab_test, compare_two_groups, binary_logistic_regression
)

# 3. Import bộ máy Xử lý Bằng chứng (PDF, PubMed, VN Journals)
from evidence_engine import (
    SourceDocument, EvidenceChunk, get_serpapi_key,
    extract_pdf, search_pubmed, search_vn_journals,
    ingest_pubmed_article, ingest_vn_article, add_source_and_chunks
)

# 4. Import bộ máy Quản lý Checkpoint / Dự án
from project_storage import save_project, load_project, list_projects, delete_project
from citation_engine import CitationEngine

# 5. Import bộ máy Xử lý Dữ liệu thô (Excel)
from data_engine import auto_clean_data

# ============================================================
# 1. CẤU HÌNH
# ============================================================
st.set_page_config(
    page_title="NCKH",
    page_icon="🔬",
    layout="wide",
)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MODEL_LITE = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

DEFAULT_TOP_K = 8
MAX_TOP_K = 20
DEFAULT_VN_JOURNAL_DOMAINS = [
    "tapchiyhocvietnam.vn",
    "vjol.info",
    "tapchinghiencuuyhoc.vn",
    "jmp.huemed-univ.edu.vn",
]

# ============================================================
# 2. CSS – GIAO DIỆN HỌC THUẬT, TỐI GIẢN
# ============================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #f8f9fa;
        color: #2c3e50;
    }

    h1 {
        color: #1e293b !important;
        text-align: center;
        font-weight: 800;
        margin-top: 0 !important;
        margin-bottom: 6px;
    }
    h2, h3, h4 { color: #0f172a !important; font-weight: 600; }

    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 32px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600;
        color: #64748b;
        transition: background-color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    div.stButton > button, div.stDownloadButton > button {
        background-color: #ffffff !important;
        color: #334155 !important;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #cbd5e1 !important;
        padding: 8px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: #94a3b8 !important;
        background-color: #f8fafc !important;
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 1px #2563eb !important;
    }

    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 6px;
        font-weight: 600;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    .stAlert { border-radius: 8px !important; }

    .warning-box { border-left: 4px solid #f59e0b; padding: 12px 16px; background: #fffbeb; border-radius: 6px; color: #92400e; font-size: 0.95rem;}
    .danger-box  { border-left: 4px solid #ef4444; padding: 12px 16px; background: #fef2f2; border-radius: 6px; color: #991b1b; font-size: 0.95rem;}
    .success-box { border-left: 4px solid #10b981; padding: 12px 16px; background: #ecfdf5; border-radius: 6px; color: #065f46; font-size: 0.95rem;}

    .stMarkdown p, .stMarkdown li {
        font-size: 1rem !important;
        line-height: 1.8 !important;
        color: #334155;
        text-align: justify !important;
    }
    .stMarkdown h3 {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 24px;
        color: #1e293b;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 4. QUẢN LÝ STATE (VERSIONED UI STATE & PERSISTENT STATE)
# ============================================================

UI_NAMESPACE = "nckh_cki"

def init_state():
    if "ui_version" not in st.session_state:
        st.session_state["ui_version"] = 0

    defaults = {
        "documents": {},
        "chunks": [],
        "embeddings": None,
        "bm25": None,
        "citation_registry": {},
        "audit_log": [],
        "last_generated": "",
        "last_evidence": [],
        "current_references": [],
        "vn_journal_domains": list(DEFAULT_VN_JOURNAL_DOMAINS),
        "t3_pm_data": [],
        "t3_vn_data": [],
        "t3_en_keyword": "",
        "t3_query": "",
        "result_cart": [],
        "saved_tables": {},
        "selection_decisions": [],
        "narrative_plan": {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

def ui_key(widget_name: str) -> str:
    version = st.session_state.get("ui_version", 0)
    return f"{UI_NAMESPACE}_v{version}_{widget_name}"

def reset_ui_state():
    current_version = st.session_state.get("ui_version", 0)
    st.session_state["ui_version"] = current_version + 1


# ============================================================
# 5. GEMINI CLIENT
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

def get_api_key() -> Optional[str]:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")

def call_gemini(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 5,
) -> Optional[str]:
    api_key = get_api_key()
    if not api_key:
        st.error("Chưa có GEMINI_API_KEY.")
        return None
        
    client = get_gemini_client(api_key)
    model_name = model or DEFAULT_MODEL

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return None
        except Exception as exc:
            error_msg = str(exc)
            if any(code in error_msg for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                if attempt < max_retries - 1:
                    wait_time = 15  
                    time.sleep(wait_time)
                else:
                    return None
            else:
                if attempt == max_retries - 1:
                    return None
                time.sleep(3)
    return None

# ============================================================
# 6. EMBEDDING MODEL
# ============================================================
@st.cache_resource
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

def get_embeddings(texts: List[str]) -> np.ndarray:
    model = load_embedding_model(DEFAULT_EMBEDDING)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)

# ============================================================
# 7. ADD PDF DOCUMENTS HELPER
# ============================================================
def add_pdf_documents(uploaded_files) -> Tuple[int, int, List[str]]:
    new_sources, new_chunks_count, errors = 0, 0, []
    new_chunks_list = [] 
    
    for uploaded_file in uploaded_files:
        try:
            source, chunks = extract_pdf(uploaded_file)
            if add_source_and_chunks(source, chunks):
                new_sources += 1
                new_chunks_count += len(chunks)
                new_chunks_list.extend(chunks)
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")

    if new_sources:
        rebuild_index(new_chunks=new_chunks_list)

    return new_sources, new_chunks_count, errors

# ============================================================
# 8. TRA CỨU TỪ KHÓA
# ============================================================
def translate_query_to_mesh(vietnamese_query: str) -> str:
    prompt = f"""Bạn là một chuyên gia tra cứu tài liệu y khoa.
Chuyển đổi tên đề tài tiếng Việt sau thành chuỗi từ khóa tiếng Anh hiệu quả nhất để tra cứu trên PubMed.
KHÔNG dùng cấu trúc MeSH chứa dấu gạch chéo. Từ gốc: "{vietnamese_query}"
Chỉ trả về chuỗi tiếng Anh."""
    text = call_gemini(prompt, model=MODEL_LITE)
    if text: return text.strip().strip('"').strip("'").replace("\n", " ")
    return vietnamese_query

def extract_vn_keywords(vietnamese_query: str) -> str:
    prompt = f"""Rút gọn tên đề tài sau thành 1-2 từ khóa quan trọng nhất bằng tiếng Việt.
Chỉ giữ lại danh từ cốt lõi. Đề tài: "{vietnamese_query}"
Chỉ trả về từ khóa rút gọn."""
    text = call_gemini(prompt, model=MODEL_LITE)
    if text: return text.strip().strip('"').strip("'").replace("\n", " ")
    return vietnamese_query

# ============================================================
# 10. INDEX / VECTOR RETRIEVAL
# ============================================================
def evidence_database_summary() -> Dict[str, Any]:
    documents = st.session_state.get("documents", {})
    chunks = st.session_state.get("chunks", [])
    by_origin_sources: Dict[str, int] = {}
    
    for meta in documents.values():
        origin = meta.get("origin", "Khác")
        by_origin_sources[origin] = by_origin_sources.get(origin, 0) + 1

    by_origin_chunks: Dict[str, int] = {}
    source_to_origin = {sid: meta.get("origin", "Khác") for sid, meta in documents.items()}
    for c in chunks:
        origin = source_to_origin.get(c.get("source_id"), "Khác")
        by_origin_chunks[origin] = by_origin_chunks.get(origin, 0) + 1

    return {
        "total_sources": len(documents),
        "total_chunks": len(chunks),
        "by_origin_sources": by_origin_sources,
        "by_origin_chunks": by_origin_chunks,
        "index_ready": st.session_state.get("embeddings") is not None,
    }

def render_evidence_database_status(context_label: str = ""):
    summary = evidence_database_summary()
    if summary["total_sources"] == 0:
        st.markdown(
            '<div class="danger-box">⚠️ <b>Evidence Database đang RỖNG.</b> '
            "Hãy nạp PDF ở Tab 1 hoặc tra cứu và bấm \"Nạp vào Evidence Database\" ở Tab 2 trước.</div>",
            unsafe_allow_html=True,
        )
        return

    origin_labels = {
        "PDF": "📄 PDF", "PubMed": "🌍 PubMed",
        "Tạp chí VN": "🇻🇳 Tạp chí VN", "Khác": "❓ Khác",
    }
    pieces = []
    for origin, count in summary["by_origin_sources"].items():
        label = origin_labels.get(origin, origin)
        n_chunks = summary["by_origin_chunks"].get(origin, 0)
        pieces.append(f"{label}: <b>{count}</b> nguồn ({n_chunks} đoạn)")

    status_html = (
        f'<div class="success-box">✅ <b>Evidence Database{" - " + context_label if context_label else ""}:</b> '
        f'{summary["total_sources"]} nguồn / {summary["total_chunks"]} đoạn bằng chứng '
        f'&nbsp;—&nbsp; {" &nbsp;|&nbsp; ".join(pieces)}'
        f'{"" if summary["index_ready"] else " &nbsp;—&nbsp; ⚠️ chưa dựng xong index"}</div>'
    )
    st.markdown(status_html, unsafe_allow_html=True)

def rebuild_index(new_chunks: List[Dict[str, Any]] = None):
    all_chunks = st.session_state.get("chunks", [])
    if not all_chunks:
        st.session_state["embeddings"] = None
        st.session_state["bm25"] = None
        return
        
    if new_chunks and st.session_state.get("embeddings") is not None:
        new_texts = [c["text"] for c in new_chunks]
        new_matrix = get_embeddings(new_texts)
        st.session_state["embeddings"] = np.vstack([st.session_state["embeddings"], new_matrix])
    else:
        texts = [c["text"] for c in all_chunks]
        st.session_state["embeddings"] = get_embeddings(texts)

    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    st.session_state["bm25"] = BM25Okapi(tokenized_corpus)

def retrieve_evidence(query: str, k: int = 8) -> List[Dict[str, Any]]:
    chunks = st.session_state.get("chunks", [])
    matrix = st.session_state.get("embeddings")
    bm25 = st.session_state.get("bm25")
    
    if not chunks or matrix is None or bm25 is None:
        return []

    query_vector = get_embeddings([query])[0]
    semantic_scores = matrix @ query_vector

    sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
    if sem_max > sem_min:
        semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min)
    else:
        semantic_scores = np.zeros_like(semantic_scores)

    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max > bm25_min:
        bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_scores = np.zeros_like(bm25_scores)

    final_scores = (0.65 * semantic_scores) + (0.35 * bm25_scores)
    k = min(k, len(chunks))
    indices = np.argsort(final_scores)[::-1][:k]

    results = []
    for idx in indices:
        item = dict(chunks[idx])
        item["score"] = float(final_scores[idx])
        results.append(item)
        
    return results

# ============================================================
# 11. CITATION ENGINE 
# ============================================================
def get_citation_engine() -> CitationEngine:
    if "citation_engine" not in st.session_state:
        st.session_state["citation_engine"] = CitationEngine()
    return st.session_state["citation_engine"]

def source_metadata(source_id: str) -> Dict[str, Any]:
    return st.session_state["documents"].get(source_id, {})

def citation_bibliography() -> str:
    refs = st.session_state.get("current_references", [])
    rows = []
    for ref in refs:
        meta = ref["metadata"]
        citation = (
            f"[{ref['vancouver_index']}] {meta.get('authors', '')}. "
            f"{meta.get('title', meta.get('file_name', 'Tài liệu chưa xác định'))}. "
            f"{meta.get('journal', '')}. {meta.get('year', '')}."
        )
        if meta.get("doi"):
            citation += f" DOI: {meta['doi']}."
        if meta.get("pmid"):
            citation += f" PMID: {meta['pmid']}."
        if meta.get("url") and meta.get("origin") == "Tạp chí VN":
            citation += f" [{meta['url']}]"
        rows.append(citation)
    return "\n".join(rows)

# ============================================================
# 12. PROMPT ENGINE – EVIDENCE ONLY
# ============================================================
BASE_SYSTEM_RULES = """
Bạn là trợ lý nghiên cứu khoa học, hỗ trợ viết luận văn Chuyên khoa cấp I ngành Dược lâm sàng.
NGUYÊN TẮC BẮT BUỘC:
1. Tài liệu được cung cấp là nguồn bằng chứng ưu tiên duy nhất.
2. Không tự tạo số liệu, p-value, OR, RR, HR, CI95%, tỷ lệ %, liều dùng hoặc cỡ mẫu.
3. Không tự tạo tên tác giả, năm, tên bài báo.
4. Mọi khẳng định phải chèn MÃ ĐỊNH DANH của tài liệu đó ngay sau câu. VD: "Tỷ lệ này là 12% [REF-001]."
5. TUYỆT ĐỐI KHÔNG tự tạo [1], [2], [3].
"""

def generate_evidence_based(
    task: str, query: str, k: int = DEFAULT_TOP_K
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    
    evidence = retrieve_evidence(query, k=k)
    if not evidence:
        return "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận.", [], []

    engine = get_citation_engine()
    evidence_context = ""

    for ev in evidence:
        meta = st.session_state["documents"].get(ev["source_id"], {})
        tag = engine.register_evidence(ev["source_id"], meta)
        table_note = f"\nGhi chú: {ev['table_hint']}" if ev.get("table_hint") else ""
        evidence_context += (
            f"\nTài liệu {tag}:\n"
            f"Nguồn: {ev['file_name']} | Trang: {ev['page']}\n"
            f"Nội dung: {ev['text']}{table_note}\n"
        )

    prompt = f"{BASE_SYSTEM_RULES}\nNHIỆM VỤ:\n{task}\nBẰNG CHỨNG:\n{evidence_context}\nYÊU CẦU: CHỈ dùng mã [REF-...] từ tài liệu."

    output = call_gemini(prompt)
    if output is None:
        return None, evidence, []

    final_text, references, invalid_tags = engine.process_vancouver_citations(output)

    st.session_state["last_generated"] = final_text
    st.session_state["last_evidence"] = evidence
    st.session_state["current_references"] = references  

    return final_text, evidence, invalid_tags

# ============================================================
# 17. KIỂM TRA NHẤT QUÁN SỐ LIỆU + TRÙNG LẶP NỘI BỘ
# ============================================================
def extract_numeric_tokens(text: str) -> List[str]:
    if not text: return []
    return re.findall(r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*%)?", text)

def parse_number(num_str: str) -> Optional[float]:
    clean_str = num_str.replace(",", ".").replace(" ", "").replace("%", "")
    try: return float(clean_str)
    except ValueError: return None

def compare_numbers_advanced(source_text: str, generated_text: str) -> Dict[str, Any]:
    source_nums = extract_numeric_tokens(source_text)
    generated_nums = extract_numeric_tokens(generated_text)
    
    source_normalized = set(x.replace(",", ".").replace(" ", "") for x in source_nums)
    generated_normalized = set(x.replace(",", ".").replace(" ", "") for x in generated_nums)
    source_floats = set(filter(None, [parse_number(x) for x in source_normalized]))

    exact_matches = []
    derived_matches = []
    warnings = []

    for gen_num in generated_normalized:
        if re.match(r"^\[\d+\]$", gen_num): continue
        if gen_num in source_normalized:
            exact_matches.append(gen_num)
        else:
            gen_val = parse_number(gen_num)
            if gen_val is not None:
                is_derived = False
                for src_val in source_floats:
                    if math.isclose(gen_val, src_val, rel_tol=1e-4) or \
                       math.isclose(gen_val, src_val * 100, rel_tol=1e-4) or \
                       math.isclose(gen_val * 100, src_val, rel_tol=1e-4):
                        is_derived = True
                        break
                if is_derived: derived_matches.append(gen_num)
                else: warnings.append(gen_num)
            else:
                warnings.append(gen_num)

    return {
        "exact_matches": sorted(exact_matches), "derived_matches": sorted(derived_matches),
        "warnings": sorted(warnings), "source_raw": sorted(source_normalized)
    }

def audit_generated_text(text: str) -> Dict[str, Any]:
    relevant_evidence = retrieve_evidence(text, k=6)
    source_text = "\n".join(e["text"] for e in relevant_evidence)
    audit = compare_numbers_advanced(source_text, text)
    return {"evidence_used": relevant_evidence, **audit}

def normalize_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.,-]", "", text)
    return text.strip()

def ngram_set(text: str, n: int = 8) -> set:
    words = normalize_for_similarity(text).split()
    if len(words) < n: return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

def internal_overlap_audit(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    target = ngram_set(text)
    if not target: return []
    results = []
    for chunk in st.session_state["chunks"]:
        other = ngram_set(chunk["text"])
        if not other: continue
        union = len(target | other)
        if union == 0: continue
        jaccard = len(target & other) / union
        if jaccard > 0:
            results.append({
                "file": chunk["file_name"], "page": chunk["page"],
                "chunk_id": chunk["chunk_id"], "similarity": round(jaccard, 4),
                "text": chunk["text"][:500],
            })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# ============================================================
# 18. XUẤT WORD
# ============================================================
def add_markdown_body_to_doc(doc: Document, body: str):
    lines = body.split("\n")
    buffer_paragraph = []
    
    def flush_paragraph():
        if buffer_paragraph:
            doc.add_paragraph(" ".join(buffer_paragraph).strip())
            buffer_paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            doc.add_heading(stripped[4:].strip(), level=3)
        elif stripped.startswith("## "):
            flush_paragraph()
            doc.add_heading(stripped[3:].strip(), level=2)
        elif stripped.startswith("# "):
            flush_paragraph()
            doc.add_heading(stripped[2:].strip(), level=1)
        elif stripped.startswith(("- ", "* ")):
            flush_paragraph()
            doc.add_paragraph(stripped[2:].strip(), style="List Bullet")
        elif stripped.startswith("|"):
            if set(stripped.replace("|", "").replace(" ", "").replace(":", "")) <= {"-"}: continue
            flush_paragraph()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            table = doc.tables[-1] if doc.tables and getattr(doc, "_last_table_open", False) else None
            if table is None:
                table = doc.add_table(rows=1, cols=len(cells))
                table.style = "Light Grid Accent 1"
                for i, c in enumerate(cells):
                    table.rows[0].cells[i].text = c
                doc._last_table_open = True
            else:
                row = table.add_row()
                for i, c in enumerate(cells):
                    if i < len(row.cells): row.cells[i].text = c
        else:
            doc._last_table_open = False
            buffer_paragraph.append(stripped)
    flush_paragraph()

def create_word_document(title: str, body: str, bibliography: str = "") -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)
    add_markdown_body_to_doc(doc, body)
    if bibliography.strip():
        doc.add_heading("Tài liệu tham khảo", level=1)
        for item in bibliography.splitlines():
            if item.strip(): doc.add_paragraph(item.strip())
    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()

# ============================================================
# 18.5. CÁC HÀM XỬ LÝ CHO TAB AUDIT
# ============================================================
def spelling_and_terminology_check(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"{BASE_SYSTEM_RULES}\nKiểm tra chính tả, thuật ngữ đoạn sau:\n{text}"
    return call_gemini(prompt, model=MODEL_LITE)

def heuristic_ai_style_score(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"{BASE_SYSTEM_RULES}\nChỉ ra các từ ngữ rập khuôn AI trong đoạn này:\n{text}"
    return call_gemini(prompt, model=MODEL_LITE)


# ============================================================
# 19. GIAO DIỆN CHÍNH
# ============================================================
def main():
    init_state()

    st.title("🔬 HỖ TRỢ NGHIÊN CỨU KHOA HỌC")
    st.caption("Evidence-Based RAG • Tra cứu đa nguồn • Citation Registry • Statistical Engine")

    tabs = st.tabs([
        "📚 1. Tài liệu (PDF)",
        "🔍 2. Tra cứu đa nguồn",
        "✍️ 3. Viết luận văn",
        "📊 4. Phân tích số liệu",
        "🔎 5. Audit",
        "⚙️ 6. Nguồn & cấu hình",
    ])

    # ------------------------------------------------------------
    # TAB 1 – TÀI LIỆU PDF
    # ------------------------------------------------------------
    with tabs[0]:
        st.header("📚 Ngân hàng tài liệu gốc (PDF)")
        render_evidence_database_status()
        
        uploaded_files = st.file_uploader(
            "Tải PDF nghiên cứu / guideline / bài báo",
            type=["pdf"], accept_multiple_files=True,
            key=ui_key("pdf_uploader")
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📥 Nạp tài liệu vào Evidence Database", type="primary"):
                if not uploaded_files:
                    st.warning("Chưa có file PDF.")
                else:
                    with st.spinner("Đang đọc PDF, tạo source registry và embedding..."):
                        ns, nc, errors = add_pdf_documents(uploaded_files)
                    st.success(f"Đã thêm {ns} tài liệu, {nc} phân đoạn bằng chứng.")
                    for err in errors: st.error(err)

        with col2:
            if st.button("🗑️ Xóa toàn bộ ngân hàng tài liệu"):
                st.session_state["documents"] = {}
                st.session_state["chunks"] = []
                st.session_state["embeddings"] = None
                st.session_state["bm25"] = None
                st.session_state["citation_registry"] = {}
                st.session_state["last_evidence"] = []
                st.success("Đã xóa dữ liệu.")
                st.rerun()

        st.write("---")
        st.subheader("Tìm bằng chứng trong toàn bộ Evidence Database")
        evidence_query = st.text_area("Nhập vấn đề cần tìm:", key=ui_key("evidence_query"))
        top_k = st.slider("Số đoạn bằng chứng", 3, MAX_TOP_K, DEFAULT_TOP_K, key=ui_key("top_k_tab1"))

        if st.button("🔎 Truy xuất bằng chứng", key="retrieve_tab1"):
            if not evidence_query.strip():
                st.warning("Nhập câu hỏi trước.")
            else:
                evidence = retrieve_evidence(evidence_query, k=top_k)
                if not evidence:
                    st.warning("Không tìm thấy bằng chứng.")
                else:
                    for ev in evidence:
                        meta = st.session_state["documents"].get(ev["source_id"], {})
                        st.markdown(f"**{ev['chunk_id']}** _( {meta.get('origin', '')} )_\nĐiểm: {ev['score']:.4f}\n\n> {ev['text']}")

    # ------------------------------------------------------------
    # TAB 2 – TRA CỨU ĐA NGUỒN
    # ------------------------------------------------------------
    with tabs[1]:
        st.header("🔍 Tra cứu đa nguồn: PubMed (Quốc tế) + Tạp chí Y học Việt Nam")
        render_evidence_database_status()
        
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            t3_query = st.text_input("Tên đề tài nghiên cứu (tiếng Việt):", key=ui_key("t3_query_input"))
        with col_btn:
            max_res = st.number_input("Số bài", min_value=2, max_value=10, value=5, key=ui_key("t3_max_res"))

        if st.button("🚀 Tra cứu song song 2 nguồn", type="primary", key="t3_btn_search"):
            if not t3_query.strip():
                st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
            else:
                with st.spinner("Đang dịch sang MeSH..."):
                    en_query = translate_query_to_mesh(t3_query)
                    st.session_state["t3_en_keyword"] = en_query
                with st.spinner("Đang tìm PubMed..."):
                    st.session_state["t3_pm_data"] = search_pubmed(en_query, max_res)
                with st.spinner("Đang tìm tạp chí Việt Nam..."):
                    vn_short = extract_vn_keywords(t3_query)
                    st.session_state["t3_vn_keyword"] = vn_short 
                    vn_results, vn_err = search_vn_journals(vn_short, max_res)
                    st.session_state["t3_vn_data"] = vn_results

        if st.session_state.get("t3_pm_data") or st.session_state.get("t3_vn_data"):
            st.write("---")
            col_vn, col_pm = st.columns(2)
            with col_vn:
                st.markdown("### 🇻🇳 Tạp chí Việt Nam")
                for i, art in enumerate(st.session_state.get("t3_vn_data", [])):
                    with st.container(border=True):
                        st.markdown(f"**{art['title']}**")
                        if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"vn_{i}")):
                            if ingest_vn_article(art):
                                rebuild_index()
                                st.success("Đã nạp.")
            with col_pm:
                st.markdown("### 🌍 PubMed")
                for i, art in enumerate(st.session_state.get("t3_pm_data", [])):
                    with st.container(border=True):
                        st.markdown(f"**{art['title']}**")
                        if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"pm_{i}")):
                            if ingest_pubmed_article(art):
                                rebuild_index()
                                st.success("Đã nạp.")

    # ------------------------------------------------------------
    # TAB 3 – VIẾT LUẬN VĂN
    # ------------------------------------------------------------
    with tabs[2]:
        st.header("✍️ Viết luận văn dựa trên bằng chứng")
        render_evidence_database_status()
        
        my_research_data = st.text_area("🌉 Số liệu của riêng anh:", height=100, key=ui_key("my_research_data"))
        ket_qua_container = st.container()

        def run_quick_task(label: str, q: str, t: str, k: int):
            with st.spinner(f"Đang soạn {label}..."):
                output, ev, inv = generate_evidence_based(t, q, k)
                if output:
                    with ket_qua_container:
                        st.write("---")
                        st.subheader(label)
                        st.markdown(output)
                        st.code(citation_bibliography(), language="text")

        c1, c2, c3 = st.columns(3)
        if c1.button("Đặt vấn đề", use_container_width=True):
            run_quick_task("Đặt vấn đề", "Dịch tễ học, cấp thiết", "Viết Đặt vấn đề", 6)
        if c2.button("Tổng quan tài liệu", use_container_width=True):
            run_quick_task("Tổng quan", "Cơ chế, khuyến cáo", "Viết Tổng quan", 8)
        if c3.button("Bàn luận số liệu", use_container_width=True):
            if not my_research_data: st.warning("Nhập số liệu trước!")
            else: run_quick_task("Bàn luận", my_research_data, f"Dữ liệu: {my_research_data}\nViết Bàn luận", 8)

    # ------------------------------------------------------------
    # TAB 4 – PHÂN TÍCH SỐ LIỆU & TUYỂN CHỌN BẢNG
    # ------------------------------------------------------------
    with tabs[3]:
        st.header("📊 Phân tích số liệu bệnh án")

        excel_file = st.file_uploader("Tải file Excel", type=["xlsx", "xls"], key=ui_key("excel_data"))

        if excel_file is not None:
            try:
                raw_df = pd.read_excel(excel_file)
                with st.spinner("Đang dọn dẹp và chuẩn hóa dữ liệu bằng Data Engine..."):
                    df, clean_logs = auto_clean_data(raw_df)
                
                st.success(f"Dữ liệu sẵn sàng: {df.shape[0]} dòng × {df.shape[1]} cột.")
                if clean_logs:
                    with st.expander("🛠️ Xem nhật ký tự động dọn dẹp dữ liệu", expanded=True):
                        for log in clean_logs: st.write(log)

                with st.expander("Xem dữ liệu sau khi chuẩn hóa"):
                    st.dataframe(df)

                # ==========================================
                # HIỂN THỊ GIỎ KẾT QUẢ VÀ TẢI VỀ WORD
                # ==========================================
                st.write("---")
                st.markdown(f"### 🛒 Giỏ kết quả: **{len(st.session_state.get('result_cart', []))}** bảng đã lưu")
                
                col_cart1, col_cart2 = st.columns(2)
                with col_cart1:
                    if st.button("🗑️ Xóa toàn bộ Giỏ kết quả", use_container_width=True):
                        st.session_state["result_cart"] = []
                        st.session_state["saved_tables"] = {}
                        st.rerun()
                
                with col_cart2:
                    saved_tabs = st.session_state.get("saved_tables", {})
                    if saved_tabs:
                        md_content = ""
                        for table_id, df_table in saved_tabs.items():
                            md_content += f"### Kết quả Thống kê: {table_id}\n\n"
                            header = "| " + " | ".join(str(c) for c in df_table.columns) + " |"
                            separator = "|" + "|".join(["---"] * len(df_table.columns)) + "|"
                            rows = ["| " + " | ".join(str(x) for x in r.values) + " |" for _, r in df_table.iterrows()]
                            md_content += "\n".join([header, separator] + rows) + "\n\n"
                        
                        docx_data = create_word_document("Phụ lục Số liệu Thống kê", md_content, "")
                        st.download_button("📥 Tải TẤT CẢ bảng ra file Word", data=docx_data, file_name="Phu_luc.docx", use_container_width=True)

                st.write("---")

                # ==========================================
                # CÁC PHÉP THỐNG KÊ 
                # ==========================================
                st.subheader("1. Thống kê mô tả")
                desc_vars = st.multiselect("Chọn biến", df.columns, key=ui_key("desc_vars"))
                if st.button("Tính tần số", key="calc_desc"):
                    for var in desc_vars:
                        result = descriptive_table(df, var)
                        if not result.empty:
                            st.dataframe(result)
                            result_id = f"DESC_{var}"
                            st.session_state["saved_tables"][result_id] = result
                            st.session_state["result_cart"].append(CandidateResult(id=result_id, title=f"Mô tả {var}", result_type="desc", variables=[var], scientific_value=3, clinical_importance=3, discussion_value=3))
                    st.success("Đã nạp vào Giỏ!")

                st.write("---")
                st.subheader("2. Bảng chéo (Chi-square)")
                c1, c2 = st.columns(2)
                deps = c1.multiselect("Biến phụ thuộc", df.columns, key=ui_key("deps"))
                indeps = c2.multiselect("Biến độc lập", df.columns, key=ui_key("indeps"))
                if st.button("Quét Crosstab", key="calc_cross"):
                    for dep in deps:
                        for indep in indeps:
                            if dep != indep:
                                res = crosstab_test(df, indep, dep)
                                st.dataframe(res["table"])
                                st.session_state["saved_tables"][f"CROSS_{indep}_{dep}"] = res["table"]
                                st.session_state["result_cart"].append(CandidateResult(id=f"CROSS_{indep}_{dep}", title=f"Cross {indep} {dep}", result_type="cross", variables=[indep, dep], scientific_value=4, clinical_importance=4, discussion_value=4))
                    st.success("Đã nạp vào Giỏ!")

                st.write("---")
                # ==========================================
                # 8. DIỄN GIẢI BẰNG AI
                # ==========================================
                st.subheader("8. Diễn giải kết quả bằng AI")
                table_options = ["-- Tự dán số liệu --"] + list(st.session_state.get("saved_tables", {}).keys())
                sel_table = st.selectbox("Chọn bảng từ Giỏ:", options=table_options, key=ui_key("sel_ai_table"))
                req = st.text_area("Dán số liệu bổ sung:", key=ui_key("req_ai"))

                if st.button("🤖 AI diễn giải", type="primary", key="ai_interpret"):
                    final_data = ""
                    if sel_table != "-- Tự dán số liệu --":
                        final_data += st.session_state["saved_tables"][sel_table].to_markdown() + "\n"
                    if req.strip(): final_data += req.strip()
                    
                    if final_data:
                        prompt = f"{BASE_SYSTEM_RULES}\nDIỄN GIẢI KẾT QUẢ SAU:\n{final_data}"
                        out = call_gemini(prompt)
                        if out: st.markdown(out)
                    else:
                        st.warning("Chưa có số liệu")

            except Exception as exc:
                st.error(f"Lỗi hệ thống Excel: {exc}")

    # ------------------------------------------------------------
    # TAB 5 & TAB 6
    # ------------------------------------------------------------
    with tabs[4]:
        st.header("🔎 Audit luận văn")
        st.info("Nhập văn bản để rà soát lỗi chính tả và check AI.")
        audit_text = st.text_area("Đoạn văn:", key=ui_key("audit_text"))
        if st.button("Check Lỗi"):
            if audit_text:
                st.write(spelling_and_terminology_check(audit_text))

    with tabs[5]:
        st.header("⚙️ Nguồn và Cấu hình")
        st.write("Phiên bản đã tối ưu hóa Data Engine và Tải Word Toàn Bộ Bảng.")
        if st.button("🗑️ Reset Hệ Thống", key="hard_reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
