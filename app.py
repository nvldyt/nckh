# app.py
# ============================================================
# HỖ TRỢ NGHIÊN CỨU KHOA HỌC – EVIDENCE-BASED RAG
# Bản tối ưu cho luận văn Chuyên khoa cấp I – Dược lâm sàng
# ============================================================

import io
import os
import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from pypdf import PdfReader

# Google Gemini SDK mới: pip install google-genai
from google import genai
from google.genai import types

# Embedding: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# DOCX
from docx import Document
from docx.shared import Pt

# ============================================================
# IMPORT TỪ CÁC MODULE ĐÃ ĐƯỢC BÓC TÁCH
# ============================================================
from table_selection_engine import (
    StudyObjective, CandidateResult,
    TableSelectionEngine, NarrativePlanner,
    Priority, Presentation
)
from statistical_engine import (
    validate_dataframe, descriptive_table, numeric_summary, 
    crosstab_test, compare_two_groups, binary_logistic_regression
)
from evidence_engine import (
    SourceDocument, EvidenceChunk, get_serpapi_key,
    extract_pdf, search_pubmed, search_vn_journals, 
    ingest_pubmed_article, ingest_vn_article, add_source_and_chunks
)
from project_storage import save_project, load_project, list_projects

# ============================================================
# 1. CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="NCKH",
    page_icon="🔬",
    layout="wide",
)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
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
# 2. CSS – GIAO DIỆN SẶC SỠ & TRONG SUỐT HEADER
# ============================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700;800&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li, .stMarkdown {
        font-family: 'Be Vietnam Pro', 'Arial', sans-serif;
    }

    /* ===== NỀN TOÀN TRANG VÀ LÀM TRONG SUỐT THANH HEADER ===== */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ff9a9e, #a18cd1, #667eea, #43e97b, #38f9d7, #6a1b9a);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Đẩy lùi nội dung xuống một chút để không bị lẹm vào Header */
    .block-container { 
        max-width: 1450px; 
        padding-top: 4.5rem !important; 
        padding-bottom: 2rem !important;
    }

    /* ===== TIÊU ĐỀ CHÍNH ===== */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 800;
        letter-spacing: 1px;
        margin-top: 0 !important;
        margin-bottom: 6px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.35), 0 0 30px rgba(255,255,255,0.25);
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        text-align: center;
    }
    h1 + div p, .stApp > div > div > div > div > div:has(h1) + div {
        color: rgba(255,255,255,0.92) !important;
    }
    h2, h3 { color: #4a148c !important; font-weight: 700; }

    /* ===== KHỐI NỘI DUNG TAB - HIỆU ỨNG KÍNH MỜ (GLASSMORPHISM) ===== */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
        border: 1px solid rgba(255,255,255,0.4);
        margin-top: 10px;
    }

    /* ===== THANH TAB ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 6px;
        gap: 6px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 700;
        color: #ffffff;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6a1b9a, #ab47bc) !important;
        color: #fff !important;
        box-shadow: 0 4px 10px rgba(106,27,154,0.4);
    }

    /* ===== NÚT BẤM - GRADIENT SẶC SỠ + HOVER ===== */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #6a1b9a 0%, #ab47bc 50%, #ff6ec4 100%) !important;
        color: white !important;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 6px 14px rgba(106,27,154,0.35);
        transition: all 0.25s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 22px rgba(106,27,154,0.5);
        filter: brightness(1.08);
    }
    div.stButton > button:active { transform: translateY(0px) scale(0.98); }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ff512f, #f09819) !important;
        box-shadow: 0 6px 14px rgba(255,81,47,0.4);
    }

    /* ===== BẢNG DỮ LIỆU ===== */
    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 14px;
        padding: 10px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }

    /* ===== Ô NHẬP LIỆU ===== */
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        border-radius: 12px !important;
        border: 1.5px solid #d1a3f0 !important;
        background-color: rgba(255,255,255,0.9) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #6a1b9a !important;
        box-shadow: 0 0 0 3px rgba(106,27,154,0.15) !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1.5px solid #d1a3f0 !important;
    }

    /* ===== EXPANDER / CONTAINER / ALERT ===== */
    .streamlit-expanderHeader {
        background: rgba(171, 71, 188, 0.12);
        border-radius: 10px;
        font-weight: 600;
        color: #4a148c;
    }
    .stAlert { border-radius: 12px !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid rgba(171,71,188,0.25) !important;
        background: rgba(255,255,255,0.7);
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
        background: rgba(255,255,255,0.6);
        padding: 10px;
    }

    /* ===== CÁC KHỐI CẢNH BÁO TÙY CHỈNH ===== */
    .source-card {
        border: 1px solid #d9e2ec; border-radius: 10px;
        padding: 10px 14px; margin-bottom: 8px; background: white;
    }
    .warning-box { border-left: 5px solid #f0ad4e; padding: 10px 14px; background: #fff8e8; border-radius: 8px; }
    .danger-box  { border-left: 5px solid #d9534f; padding: 10px 14px; background: #fff1f0; border-radius: 8px; }
    .success-box { border-left: 5px solid #2e8b57; padding: 10px 14px; background: #eef9f1; border-radius: 8px; }

    /* ===== VĂN BẢN HỌC THUẬT DO AI TẠO - DỄ ĐỌC, CANH ĐỀU 2 LỀ ===== */
    .stMarkdown p, .stMarkdown li {
        font-size: 0.95rem !important;
        line-height: 1.75 !important;
        text-align: justify !important;
        text-justify: inter-word;
    }
    .stMarkdown table td, .stMarkdown table th { font-size: 0.85rem !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        text-align: left !important;
        border-left: 5px solid #ab47bc;
        padding-left: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "documents": {},
        "chunks": [],
        "embeddings": None,
        "citation_registry": {},
        "audit_log": [],
        "last_generated": "",
        "last_evidence": [],
        "vn_journal_domains": list(DEFAULT_VN_JOURNAL_DOMAINS),
        "t3_pm_data": [],
        "t3_vn_data": [],
        "t3_en_keyword": "",
        "t3_query": "",
        "structure_locked": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()


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
        st.error(
            "Chưa có GEMINI_API_KEY. Hãy thêm vào Streamlit Secrets "
            "hoặc biến môi trường."
        )
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
                    status = st.warning(f"⏳ Trạm máy chủ Google đang quá tải đột xuất. Ứng dụng tự động đợi {wait_time} giây rồi thử lại (Lần {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    status.empty()
                else:
                    st.error("❌ Máy chủ Google Gemini hiện đang quá bận. Anh vui lòng đợi 1-2 phút rồi bấm thử lại nhé!")
                    return None
            else:
                if attempt == max_retries - 1:
                    st.error(f"Lỗi Gemini: {error_msg}")
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
    new_sources, new_chunks, errors = 0, 0, []

    for uploaded_file in uploaded_files:
        try:
            source, chunks = extract_pdf(uploaded_file)
            if add_source_and_chunks(source, chunks):
                new_sources += 1
                new_chunks += len(chunks)
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")

    if new_sources:
        rebuild_index()

    return new_sources, new_chunks, errors


# ============================================================
# 8. TRA CỨU MESH (Helper)
# ============================================================

def translate_query_to_mesh(vietnamese_query: str) -> str:
    prompt = (
        "Chuyển đổi từ khóa tiếng Việt sau thành chuỗi từ khóa y khoa "
        "(MeSH terms) bằng tiếng Anh tối ưu nhất để tìm trên PubMed.\n"
        f"Từ gốc: {vietnamese_query}\n"
        "Chỉ trả về chuỗi tiếng Anh, không giải thích, không markdown."
    )
    text = call_gemini(prompt, model=MODEL_LITE, temperature=0.1)
    if text:
        return text.strip().strip('"').strip("'")
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
            "Chưa có tài liệu nào được nạp — các nút viết nhanh sẽ trả về "
            '"chưa đủ bằng chứng để kết luận". Hãy nạp PDF ở Tab 1 hoặc tra cứu '
            "và bấm \"Nạp vào Evidence Database\" ở Tab 2 trước.</div>",
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
        f'{"" if summary["index_ready"] else " &nbsp;—&nbsp; ⚠️ chưa dựng xong index, thử tải lại"}</div>'
    )
    st.markdown(status_html, unsafe_allow_html=True)


def rebuild_index():
    chunks = st.session_state["chunks"]
    if not chunks:
        st.session_state["embeddings"] = None
        return
    texts = [c["text"] for c in chunks]
    st.session_state["embeddings"] = get_embeddings(texts)


def retrieve_evidence(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    chunks = st.session_state["chunks"]
    matrix = st.session_state.get("embeddings")

    if not chunks or matrix is None:
        return []

    query_vector = get_embeddings([query])[0]
    scores = matrix @ query_vector

    k = min(k, len(chunks))
    indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in indices:
        item = dict(chunks[idx])
        item["score"] = float(scores[idx])
        results.append(item)
    return results


def format_evidence_for_prompt(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "KHÔNG CÓ BẰNG CHỨNG ĐƯỢC TRUY XUẤT."

    blocks = []
    for ev in evidence:
        meta = st.session_state["documents"].get(ev["source_id"], {})
        origin_note = f" | Nguồn gốc: {meta.get('origin', '')}"
        table_note = f"\nGhi chú: {ev['table_hint']}" if ev.get("table_hint") else ""

        blocks.append(
            f"""
[SOURCE_TAG={ev['chunk_id']}]
Nguồn: {ev['file_name']}{origin_note}
Trang/mục: {ev['page'] if ev['page'] else ev.get('section', '')}
Source ID: {ev['source_id']}
Điểm tương đồng: {ev.get('score', 0):.4f}{table_note}

NỘI DUNG GỐC:
{ev['text']}
[/SOURCE_TAG]
""".strip()
        )

    return "\n\n".join(blocks)


# ============================================================
# 11. CITATION ENGINE
# ============================================================

def source_metadata(source_id: str) -> Dict[str, Any]:
    return st.session_state["documents"].get(source_id, {})


def register_citations(evidence: List[Dict[str, Any]]) -> Dict[str, int]:
    registry = st.session_state["citation_registry"]
    for ev in evidence:
        source_id = ev["source_id"]
        if source_id not in registry:
            registry[source_id] = len(registry) + 1
    return registry


def citation_bibliography() -> str:
    registry = st.session_state["citation_registry"]
    rows = []
    for source_id, number in sorted(registry.items(), key=lambda x: x[1]):
        meta = source_metadata(source_id)
        citation = (
            f"[{number}] {meta.get('authors', '')}. "
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


def replace_source_tags_with_citations(
    generated_text: str, evidence: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    registry = register_citations(evidence)
    valid_chunk_to_source = {ev["chunk_id"]: ev["source_id"] for ev in evidence}
    invalid_tags = []

    pattern = re.compile(r"\[SOURCE_TAG=([A-Za-z0-9_-]+)\]")

    def repl(match):
        chunk_id = match.group(1)
        if chunk_id not in valid_chunk_to_source:
            invalid_tags.append(chunk_id)
            return "[CITATION_INVALID]"
        source_id = valid_chunk_to_source[chunk_id]
        return f"[{registry[source_id]}]"

    converted = pattern.sub(repl, generated_text)
    return converted, invalid_tags


# ============================================================
# 12. PROMPT ENGINE – EVIDENCE ONLY
# ============================================================

BASE_SYSTEM_RULES = """
Bạn là trợ lý nghiên cứu khoa học, hỗ trợ viết luận văn Chuyên khoa cấp I
ngành Dược lâm sàng.

NGUYÊN TẮC BẮT BUỘC:
1. Tài liệu được cung cấp là nguồn bằng chứng ưu tiên duy nhất.
2. Không tự tạo số liệu, p-value, OR, RR, HR, CI95%, tỷ lệ %, liều dùng hoặc cỡ mẫu.
3. Không tự tạo tên tác giả, năm, tên bài báo, DOI, PMID.
4. Mọi khẳng định phải gắn SOURCE_TAG thật ngay sau khẳng định: [SOURCE_TAG=SRC-...-Pxxx-Cxxx]
5. Văn phong chuyên khảo Dược lâm sàng, khô khan, trực diện, không hoa mỹ.
"""


def generate_evidence_based(
    task: str, query: str, k: int = DEFAULT_TOP_K
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    evidence = retrieve_evidence(query, k=k)

    if not evidence:
        return "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận.", [], []

    evidence_text = format_evidence_for_prompt(evidence)

    prompt = f"""
{BASE_SYSTEM_RULES}

NHIỆM VỤ:
{task}

CÂU HỎI/TRUY VẤN:
{query}

BẰNG CHỨNG ĐƯỢC PHÉP SỬ DỤNG:
{evidence_text}
"""

    output = call_gemini(prompt)
    if output is None:
        return None, evidence, []

    converted, invalid_tags = replace_source_tags_with_citations(output, evidence)

    if invalid_tags:
        converted += (
            "\n\n> ⚠️ CẢNH BÁO AUDIT: Có SOURCE_TAG không tồn tại trong bằng "
            "chứng được truy xuất. Đoạn này cần kiểm tra thủ công trước khi sử dụng."
        )

    st.session_state["last_generated"] = converted
    st.session_state["last_evidence"] = evidence
    return converted, evidence, invalid_tags


# ============================================================
# 17. KIỂM TRA NHẤT QUÁN SỐ LIỆU + TRÙNG LẶP NỘI BỘ
# ============================================================

def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*%)?"
    return re.findall(pattern, text)


def compare_numbers(source_text: str, generated_text: str) -> Dict[str, Any]:
    source_nums = extract_numeric_tokens(source_text)
    generated_nums = extract_numeric_tokens(generated_text)

    source_normalized = set(x.replace(",", ".").replace(" ", "") for x in source_nums)
    generated_normalized = set(x.replace(",", ".").replace(" ", "") for x in generated_nums)
    suspicious = sorted(generated_normalized - source_normalized)

    return {
        "source_numbers": sorted(source_normalized),
        "generated_numbers": sorted(generated_normalized),
        "suspicious_generated_numbers": suspicious,
    }


def audit_generated_text(text: str) -> Dict[str, Any]:
    evidence = st.session_state.get("last_evidence", [])
    source_text = "\n".join(e["text"] for e in evidence)
    audit = compare_numbers(source_text, text)
    citation_invalid = "[CITATION_INVALID]" in text
    return {"invalid_citation": citation_invalid, **audit}


def normalize_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.,-]", "", text)
    return text.strip()


def ngram_set(text: str, n: int = 8) -> set:
    words = normalize_for_similarity(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def internal_overlap_audit(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    target = ngram_set(text)
    if not target:
        return []

    results = []
    for chunk in st.session_state["chunks"]:
        other = ngram_set(chunk["text"])
        if not other:
            continue

        intersection = len(target & other)
        union = len(target | other)
        if union == 0:
            continue

        jaccard = intersection / union
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
            if set(stripped.replace("|", "").replace(" ", "").replace(":", "")) <= {"-"}:
                continue
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
                    if i < len(row.cells):
                        row.cells[i].text = c
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
            if item.strip():
                doc.add_paragraph(item.strip())

    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


# ============================================================
# 18.5. HÀM AUDIT (NGÔN NGỮ, ĐẠO VĂN, AI-STYLE)
# ============================================================

def spelling_and_terminology_check(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"""
    {BASE_SYSTEM_RULES}
    Bạn là biên tập viên y khoa chuyên ngành Dược lâm sàng. Rà soát lỗi chính tả và thuật ngữ y khoa trong đoạn sau:
    {text}
    """
    return call_gemini(prompt, model=MODEL_LITE)

def plagiarism_style_review(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"""
    {BASE_SYSTEM_RULES}
    Đóng vai hội đồng phản biện luận văn CKI Dược lâm sàng, đánh giá tính logic và văn phong học thuật:
    {text}
    """
    return call_gemini(prompt)

def heuristic_ai_style_score(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"""
    {BASE_SYSTEM_RULES}
    Phân tích các dấu hiệu văn bản có khả năng do AI viết trong đoạn sau:
    {text}
    """
    return call_gemini(prompt, model=MODEL_LITE)


# ============================================================
# 19. GIAO DIỆN CHÍNH (FULL TÍNH NĂNG)
# ============================================================

st.title("🔬 HỖ TRỢ NGHIÊN CỨU KHOA HỌC")
st.caption(
    "Evidence-Based RAG • Tra cứu đa nguồn (PubMed + Tạp chí VN) • "
    "Citation Registry • Statistical Engine • Audit"
)

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
                for err in errors:
                    st.error(err)

    with col2:
        if st.button("🗑️ Xóa toàn bộ ngân hàng tài liệu"):
            st.session_state["documents"] = {}
            st.session_state["chunks"] = []
            st.session_state["embeddings"] = None
            st.session_state["citation_registry"] = {}
            st.session_state["last_evidence"] = []
            st.success("Đã xóa dữ liệu trong phiên hiện tại.")
            st.rerun()

    st.write("---")
    st.subheader("Nguồn đã nạp (tất cả nguồn gốc)")

    docs = list(st.session_state["documents"].values())
    if docs:
        st.dataframe(pd.DataFrame(docs))
    else:
        st.info("Chưa có tài liệu.")

    st.subheader("Tìm bằng chứng trong toàn bộ Evidence Database")

    evidence_query = st.text_area(
        "Nhập vấn đề cần tìm trong tài liệu:",
        placeholder=(
            "Ví dụ: tỷ lệ bệnh nhân đạt huyết áp mục tiêu hoặc "
            "tiêu chuẩn hiệu chỉnh liều theo eGFR"
        ),
    )
    top_k = st.slider("Số đoạn bằng chứng", 3, MAX_TOP_K, DEFAULT_TOP_K, key="tk1")

    if st.button("🔎 Truy xuất bằng chứng", key="retrieve_tab1"):
        if not evidence_query.strip():
            st.warning("Nhập câu hỏi trước.")
        else:
            evidence = retrieve_evidence(evidence_query, k=top_k)
            st.session_state["last_evidence"] = evidence

            if not evidence:
                st.warning("Không tìm thấy bằng chứng trong tài liệu.")
            else:
                for ev in evidence:
                    meta = st.session_state["documents"].get(ev["source_id"], {})
                    st.markdown(
                        f"""**{ev['chunk_id']}** _( {meta.get('origin', '')} )_
Nguồn: `{ev['file_name']}` — Trang/mục: **{ev['page'] or ev.get('section', '')}**
Điểm tương đồng: **{ev['score']:.4f}**

> {ev['text']}
"""
                    )


# ------------------------------------------------------------
# TAB 2 – TRA CỨU ĐA NGUỒN: PUBMED + TẠP CHÍ Y HỌC VIỆT NAM
# ------------------------------------------------------------
with tabs[1]:
    st.header("🔍 Tra cứu đa nguồn: PubMed (Quốc tế) + Tạp chí Y học Việt Nam")
    st.info(
        "Nhập tên đề tài bằng tiếng Việt. Hệ thống tự dịch sang từ khoá MeSH để "
        "tìm trên PubMed, đồng thời tìm bài báo tiếng Việt liên quan."
    )
    render_evidence_database_status()

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        t3_query = st.text_input(
            "Tên đề tài nghiên cứu (tiếng Việt):",
            placeholder="VD: Hiệu quả kiểm soát đường huyết bằng metformin...",
            key="t3_query_input",
        )
    with col_btn:
        max_res = st.number_input("Số bài/nguồn", min_value=2, max_value=10, value=5, key="t3_max_res")

    if st.button("🚀 Tra cứu song song 2 nguồn", type="primary", key="t3_btn_search"):
        if not t3_query.strip():
            st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
        else:
            st.session_state["t3_query"] = t3_query
            with st.spinner("Đang dịch & chuẩn hoá từ khoá sang MeSH..."):
                en_query = translate_query_to_mesh(t3_query)
                st.session_state["t3_en_keyword"] = en_query
            with st.spinner("Đang tìm & tải Abstract từ PubMed..."):
                st.session_state["t3_pm_data"] = search_pubmed(en_query, max_res)
            with st.spinner("Đang tìm bài báo trên tạp chí Y học Việt Nam..."):
                vn_results, vn_err = search_vn_journals(t3_query, max_res)
                st.session_state["t3_vn_data"] = vn_results
                if vn_err:
                    st.warning(vn_err)

    if st.session_state["t3_pm_data"] or st.session_state["t3_vn_data"]:
        st.write("---")
        col_vn, col_pm = st.columns(2)

        with col_vn:
            st.markdown("### 🇻🇳 Tạp chí Y học Việt Nam")
            if not st.session_state["t3_vn_data"]:
                st.info("Chưa có dữ liệu.")
            else:
                for i, art in enumerate(st.session_state["t3_vn_data"]):
                    with st.container(border=True):
                        st.markdown(f"**[{art['title']}]({art['link']})**" if art.get("link") else f"**{art['title']}**")
                        st.caption(art["source"])
                        st.write(art["snippet"])
                        if st.button("➕ Nạp vào Evidence Database", key=f"vn_ingest_{i}"):
                            if ingest_vn_article(art):
                                rebuild_index()
                                st.success("Đã nạp thành công.")
                            else:
                                st.info("Nguồn đã tồn tại.")

        with col_pm:
            st.markdown("### 🌍 PubMed (Quốc tế)")
            if st.session_state["t3_en_keyword"]:
                st.success(f"🔑 MeSH: **{st.session_state['t3_en_keyword']}**")
            if not st.session_state["t3_pm_data"]:
                st.info("Chưa có dữ liệu.")
            else:
                for i, art in enumerate(st.session_state["t3_pm_data"]):
                    with st.container(border=True):
                        st.markdown(f"**[{art['title']}]({art['url']})**")
                        st.caption(f"✍️ {art['authors']} ({art['year']}) — {art['journal']}")
                        with st.expander("Xem tóm tắt"):
                            st.write(art["abstract"])
                        if st.button("➕ Nạp vào Evidence Database", key=f"pm_ingest_{i}"):
                            if ingest_pubmed_article(art):
                                rebuild_index()
                                st.success("Đã nạp thành công.")
                            else:
                                st.info("Nguồn đã tồn tại.")

        st.write("---")
        if st.button("➕ Nạp TẤT CẢ kết quả ở trên", key="t3_ingest_all"):
            count = 0
            for art in st.session_state["t3_pm_data"]:
                if ingest_pubmed_article(art): count += 1
            for art in st.session_state["t3_vn_data"]:
                if ingest_vn_article(art): count += 1
            if count:
                rebuild_index()
            st.success(f"Đã nạp {count} nguồn mới.")


# ------------------------------------------------------------
# TAB 3 – VIẾT LUẬN VĂN (FULL NÚT VIẾT NHANH)
# ------------------------------------------------------------
with tabs[2]:
    st.header("✍️ Viết luận văn dựa trên bằng chứng")
    st.warning("Công cụ tạo bản nháp. Cần đối chiếu bản gốc trước khi đưa vào luận văn chính thức.")
    render_evidence_database_status("cho các nút viết nhanh bên dưới")

    my_research_data = st.text_area(
        "🌉 Số liệu nghiên cứu của riêng anh (dùng cho Bàn luận / So sánh):",
        placeholder="VD: 'Tỷ lệ nam/nữ là 1.42:1' hoặc dán kết quả từ Tab 4...",
        height=140,
    )

    citation_rules = """
QUY TẮC TRÍCH DẪN & HÀN LÂM BẮT BUỘC:
1. Chỉ dùng SOURCE_TAG thật để hệ thống tự chuyển thành [n].
2. Các số trích dẫn theo đúng thứ tự xuất hiện.
3. Không dùng kiểu [Tên tác giả, Năm].
4. Dùng Heading 3 (###) cho tiêu đề mục.
"""

    ket_qua_container = st.container()

    def run_quick_task(task_label: str, query: str, task_prompt: str, k: int):
        with st.spinner(f"AI đang soạn: {task_label}..."):
            output, evidence, invalid = generate_evidence_based(task_prompt, query, k=k)

        if output:
            with ket_qua_container:
                st.write("---")
                st.subheader(task_label)
                st.markdown(output)

                bib = citation_bibliography()
                with st.expander("📖 Tài liệu tham khảo đã đăng ký (toàn phiên)"):
                    st.code(bib if bib else "Chưa có citation registry.", language="text")

                audit = audit_generated_text(output)
                colA, colB = st.columns(2)
                with colA:
                    if audit["invalid_citation"]:
                        st.error("Có citation không hợp lệ.")
                    else:
                        st.success("Citation hợp lệ.")
                with colB:
                    if audit["suspicious_generated_numbers"]:
                        st.warning(f"Số liệu lạ: {audit['suspicious_generated_numbers']}")
                    else:
                        st.success("Số liệu khớp bằng chứng.")

    st.subheader("📝 Lệnh viết nhanh")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1: btn_dat_van_de = st.button("Đặt vấn đề", use_container_width=True)
    with c2: btn_tong_quan = st.button("Tổng quan tài liệu", use_container_width=True)
    with c3: btn_phuong_phap = st.button("Phương pháp NC", use_container_width=True)
    with c4: btn_ban_luan = st.button("Bàn luận toàn diện", use_container_width=True)
    with c5: btn_so_sanh = st.button("So sánh NC liên quan", use_container_width=True)
    with c6: btn_tltk = st.button("Trích dẫn TLTK", use_container_width=True)

    st.write("---")
    st.subheader("Lệnh tùy chỉnh")
    custom_prompt = st.text_area("Nhập câu lệnh khác:", key="custom_prompt_tab3")
    k_custom = st.slider("Số nguồn bằng chứng truy xuất", 3, MAX_TOP_K, DEFAULT_TOP_K, key="tk3")
    btn_custom = st.button("▶️ Chạy lệnh tùy chỉnh")

    if btn_dat_van_de:
        query = "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, dịch tễ học, gánh nặng bệnh tật liên quan sử dụng thuốc"
        task = f"Viết phần 'Đặt vấn đề' cho luận văn CKI Dược lâm sàng.\n{citation_rules}"
        run_quick_task("Đặt vấn đề", query, task, k=6)

    if btn_tong_quan:
        query = "Tổng quan y văn, các nghiên cứu liên quan, cơ chế dược lý, kết quả chính, khuyến cáo điều trị"
        task = f"Viết phần 'Tổng quan tài liệu' chuyên sâu.\n{citation_rules}"
        run_quick_task("Tổng quan tài liệu", query, task, k=8)

    if btn_phuong_phap:
        query = "Đối tượng nghiên cứu, tiêu chuẩn chọn loại, thiết kế nghiên cứu, cỡ mẫu, biến số nghiên cứu, phương pháp thu thập số liệu"
        task = f"Viết 'Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU'.\n{citation_rules}"
        run_quick_task("Phương pháp nghiên cứu", query, task, k=5)

    if btn_ban_luan:
        if not my_research_data.strip():
            st.warning("Cần nhập số liệu của anh vào ô 'Số liệu nghiên cứu' trước!")
        else:
            task = f"KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:\n{my_research_data}\n\nYÊU CẦU: Viết Bàn luận toàn diện, đối chiếu trực tiếp với bằng chứng.\n{citation_rules}"
            run_quick_task("Bàn luận toàn diện", my_research_data, task, k=8)

    if btn_so_sanh:
        if not my_research_data.strip():
            st.warning("Cần nhập số liệu của anh vào ô 'Số liệu nghiên cứu' trước!")
        else:
            task = f"KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:\n{my_research_data}\n\nYÊU CẦU: So sánh với các nghiên cứu và khuyến cáo liên quan.\n{citation_rules}"
            run_quick_task("So sánh nghiên cứu liên quan", my_research_data, task, k=8)

    if btn_tltk:
        query = "Tài liệu tham khảo, tác giả, năm xuất bản, tạp chí"
        task = f"Liệt kê các SOURCE_TAG phù hợp làm tài liệu tham khảo chính.\n{citation_rules}"
        with st.spinner("AI đang soạn: Trích dẫn TLTK..."):
            output, evidence, invalid = generate_evidence_based(task, query, k=10)
            if output:
                with ket_qua_container:
                    st.write("---")
                    st.subheader("Danh mục Tài liệu tham khảo")
                    bib = citation_bibliography()
                    st.code(bib if bib else "Chưa có citation registry.", language="text")

    if btn_custom:
        if not custom_prompt.strip():
            st.warning("Vui lòng nhập yêu cầu!")
        else:
            task = f"{custom_prompt}\n{citation_rules}"
            run_quick_task("Kết quả lệnh tùy chỉnh", custom_prompt, task, k=k_custom)


# ------------------------------------------------------------
# TAB 4 – PHÂN TÍCH SỐ LIỆU & RESULT DATABASE (KHÔNG LỌC P<0.05)
# ------------------------------------------------------------
with tabs[3]:
    st.header("📊 Phân tích số liệu bệnh án & Xây dựng Chương 3")

    if "result_cart" not in st.session_state:
        st.session_state["result_cart"] = []
    if "saved_tables" not in st.session_state:
        st.session_state["saved_tables"] = {}

    excel_file = st.file_uploader("Tải file Excel số liệu bệnh án", type=["xlsx", "xls"], key="excel_data")

    if excel_file is not None:
        try:
            df = pd.read_excel(excel_file)
            st.success(f"Dữ liệu: {df.shape[0]} dòng × {df.shape[1]} cột.")
            for warn in validate_dataframe(df):
                st.warning(warn)

            with st.expander("Xem dữ liệu thô"):
                st.dataframe(df)

            st.write("---")
            st.markdown(f"### 🛒 Result Database (Giỏ kết quả): **{len(st.session_state['result_cart'])}** bảng đã lưu")
            if st.button("🗑️ Xóa toàn bộ Result Database"):
                st.session_state["result_cart"] = []
                st.session_state["saved_tables"] = {}
                st.session_state.pop("selection_decisions", None)
                st.rerun()

            st.write("---")
            st.subheader("1. Thống kê mô tả (Biến phân loại)")
            all_cols = df.columns.tolist()
            desc_vars = st.multiselect("Chọn biến phân loại", all_cols, key="desc_vars")
            if st.button("Tính tần số & Nạp vào Result Database"):
                for var in desc_vars:
                    res = descriptive_table(df, var)
                    if not res.empty:
                        rid = f"DESC_{var}"
                        st.session_state["saved_tables"][rid] = res
                        if not any(c.id == rid for c in st.session_state["result_cart"]):
                            st.session_state["result_cart"].append(
                                CandidateResult(
                                    id=rid, title=f"Đặc điểm phân bố của biến {var}",
                                    result_type="demographic", variables=[var],
                                    scientific_value=4.0, clinical_importance=4.0, discussion_value=3.5
                                )
                            )
                st.success("Đã nạp các bảng mô tả vào kho!")

            st.subheader("2. Biến định lượng (Mean/SD, Median/IQR)")
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            num_vars = st.multiselect("Chọn biến định lượng", num_cols, key="num_vars")
            if st.button("Tính chỉ số định lượng & Nạp vào Result Database"):
                for var in num_vars:
                    summ = numeric_summary(df, var)
                    if summ:
                        ndf = pd.DataFrame([{"N": summ['n'], "Mean±SD": f"{summ['mean']:.2f}±{summ['sd']:.2f}", "Median(IQR)": f"{summ['median']:.2f} ({summ['q1']:.2f}-{summ['q3']:.2f})"}])
                        rid = f"NUM_{var}"
                        st.session_state["saved_tables"][rid] = ndf
                        if not any(c.id == rid for c in st.session_state["result_cart"]):
                            st.session_state["result_cart"].append(
                                CandidateResult(
                                    id=rid, title=f"Đặc điểm định lượng biến {var}",
                                    result_type="baseline", variables=[var],
                                    scientific_value=4.0, clinical_importance=4.0, discussion_value=3.5
                                )
                            )
                st.success("Đã nạp bảng định lượng vào kho!")

            st.subheader("3. Bảng chéo & Kiểm định (Chi-square / Fisher)")
            c_dep = st.multiselect("Biến phụ thuộc", all_cols, key="cdep")
            c_indep = st.multiselect("Biến độc lập", all_cols, key="cind")
            if st.button("Quét toàn bộ Bảng chéo (Lưu toàn bộ vào Result Database)"):
                for dep in c_dep:
                    for indep in c_indep:
                        if dep != indep:
                            try:
                                r = crosstab_test(df, indep, dep)
                                rid = f"CROSS_{indep}_{dep}"
                                st.session_state["saved_tables"][rid] = r["table"]
                                if not any(c.id == rid for c in st.session_state["result_cart"]):
                                    st.session_state["result_cart"].append(
                                        CandidateResult(
                                            id=rid, title=f"Mối liên quan giữa {indep} và {dep}",
                                            result_type="association", variables=[indep, dep],
                                            p_value=r['p_value'],
                                            scientific_value=4.5, clinical_importance=4.5, discussion_value=4.5
                                        )
                                    )
                            except: pass
                st.success("Đã lưu toàn bộ kết quả bảng chéo vào Result Database!")

            st.subheader("4. Hồi quy Logistic Nhị Phân (Lưu toàn bộ mô hình)")
            l_out = st.selectbox("Biến kết cục (Outcome)", all_cols, key="lout")
            l_preds = st.multiselect("Các biến dự báo (Predictors)", all_cols, key="lpred")
            if st.button("Chạy Logistic & Lưu toàn bộ mô hình"):
                if l_out and l_preds:
                    try:
                        res_df, _ = binary_logistic_regression(df, l_out, l_preds)
                        rid = f"LOG_{l_out}"
                        st.session_state["saved_tables"][rid] = res_df
                        if not any(c.id == rid for c in st.session_state["result_cart"]):
                            st.session_state["result_cart"].append(
                                CandidateResult(
                                    id=rid, title=f"Mô hình hồi quy đa biến cho kết cục: {l_out}",
                                    result_type="regression", variables=[l_out] + l_preds,
                                    scientific_value=5.0, clinical_importance=5.0, discussion_value=5.0
                                )
                            )
                        st.success("Đã lưu toàn bộ kết quả hồi quy vào Result Database!")
                    except Exception as e:
                        st.error(f"Lỗi chạy hồi quy: {e}")

            st.write("---")
            st.subheader("📋 Bộ máy tuyển chọn & Khóa cấu trúc Chương 3")
            
            with st.expander("🎯 Khai báo Mục tiêu nghiên cứu"):
                obj_1 = st.text_input("Mục tiêu 1", value="ĐẶC ĐIỂM BỆNH NHÂN NGHIÊN CỨU")
                obj_2 = st.text_input("Mục tiêu 2", value="PHÂN TÍCH THỰC TRẠNG SỬ DỤNG THUỐC")
                objectives = [
                    StudyObjective(id="MT1", title=obj_1, keywords=["tuổi", "tuoi", "giới", "gioi", "bệnh", "benh", "đặc điểm"]),
                    StudyObjective(id="MT2", title=obj_2, keywords=["thuốc", "thuoc", "phù hợp", "phu hop", "liều", "lieu", "chỉ định"]),
                ]

            if st.button("🚀 Chạy Table Selection Engine", type="primary"):
                if not st.session_state["result_cart"]:
                    st.error("Result Database đang trống!")
                else:
                    engine = TableSelectionEngine(objectives, st.session_state["result_cart"])
                    decisions = engine.run()
                    st.session_state["selection_decisions"] = decisions
                    st.success("Đã hoàn tất sắp xếp cấu trúc!")

            if "selection_decisions" in st.session_state:
                st.write("### 📊 Đề xuất Cấu trúc Chương Kết quả")
                for d in st.session_state["selection_decisions"]:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"**[{d.priority.value}] {d.title}** *(Điểm: {d.total_score:.1f})*")
                        st.caption(f"Lý do: {d.reason}")
                    with col2:
                        st.write(f"Thứ tự: {d.recommended_order or 'Phụ lục'}")
                    with col3:
                        st.checkbox("Dùng", value=True, key=f"use_{d.result_id}")

                if not st.session_state["structure_locked"]:
                    if st.button("🔒 KHÓA CẤU TRÚC CHƯƠNG 3", type="primary"):
                        st.session_state["structure_locked"] = True
                        st.success("Đã khóa cấu trúc thành công! AI sẽ tuân thủ nghiêm ngặt cấu trúc này.")
                        st.rerun()
                else:
                    st.success("🔒 Cấu trúc Chương 3 đã được KHÓA. Sẵn sàng viết luận văn.")
                    if st.button("🔓 Mở khóa để chỉnh sửa"):
                        st.session_state["structure_locked"] = False
                        st.rerun()

                st.write("### 🖨️ Bảng số liệu sẵn sàng đưa vào Word")
                for d in st.session_state["selection_decisions"]:
                    if d.result_id in st.session_state["saved_tables"]:
                        st.markdown(f"**Bảng {d.recommended_order or '*'}. {d.title}**")
                        st.markdown(st.session_state["saved_tables"][d.result_id].to_html(index=False, border=1), unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Lỗi đọc file Excel: {exc}")


# ------------------------------------------------------------
# TAB 5 – AUDIT (SỬA LỖI OVERLAPS)
# ------------------------------------------------------------
with tabs[4]:
    st.header("🔎 Audit luận văn")
 
    st.markdown(
        '<div class="warning-box">⚠️ <b>Giới hạn cần biết:</b> các công cụ ở '
        "tab này chỉ đưa ra <b>chỉ báo nguy cơ / gợi ý kiểm tra thêm</b>. "
        "Không có công cụ nào (kể cả phần mềm thương mại) khẳng định chắc "
        "chắn 100% một đoạn văn <i>không đạo văn</i> hay <i>không do AI "
        "viết</i>. Đạo văn chỉ được đối chiếu với các nguồn đã nạp trong "
        "Evidence Database, không phải toàn bộ internet.</div>",
        unsafe_allow_html=True,
    )
 
    audit_text = st.text_area("Dán đoạn văn cần kiểm tra", height=280, key="audit_text")
 
    st.write("---")
    ket_qua_audit_container = st.container()
 
    st.subheader("Nhóm 1: Đối chiếu với bằng chứng & citation")
    a1, a2, a3 = st.columns(3)
 
    with a1:
        if st.button("🔢 Audit số liệu", key="audit_numbers"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                result = audit_generated_text(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🔢 Kết quả Audit Số liệu")
                    st.write("**Số xuất hiện trong nguồn:**")
                    st.write(result["source_numbers"])
                    st.write("**Số xuất hiện trong bản viết:**")
                    st.write(result["generated_numbers"])
                    if result["suspicious_generated_numbers"]:
                        st.error("Có số cần kiểm tra:")
                        st.write(result["suspicious_generated_numbers"])
                    else:
                        st.success("Không phát hiện số mới ngoài tập bằng chứng.")
 
    with a2:
        if st.button("📚 Audit citation", key="audit_citation"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                invalid = "[CITATION_INVALID]" in audit_text
                with ket_qua_audit_container:
                    st.markdown("### 📚 Kết quả Audit Citation")
                    if invalid:
                        st.error("Có citation invalid.")
                    else:
                        st.success("Không phát hiện citation invalid.")
 
    with a3:
        if st.button("🔍 Tìm trùng lặp nội bộ", key="audit_overlap"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                overlaps = internal_overlap_audit(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🔍 Kết quả tìm trùng lặp nội bộ")
                    if not overlaps:
                        st.info("Không tìm thấy đoạn trùng đáng kể.")
                    else:
                        for item in overlaps:
                            st.markdown(
                                f"""**{item['file']} – trang {item['page']}**
Similarity nội bộ: **{item['similarity']}**
 
> {item['text']}
"""
                            )
 
    st.write("---")
    st.subheader("Nhóm 2: Ngôn ngữ, đạo văn diện rộng & chỉ báo AI-viết")
    b1, b2, b3 = st.columns(3)
 
    with b1:
        if st.button("🔤 Kiểm tra chính tả & thuật ngữ", key="audit_spelling"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("AI đang rà soát chính tả & thuật ngữ..."):
                    response = spelling_and_terminology_check(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🔤 Kết quả kiểm tra Chính tả & Thuật ngữ")
                    if response: st.markdown(response)
                    else: st.error("Không nhận được kết quả từ AI.")
 
    with b2:
        if st.button("📄 Kiểm tra nguy cơ đạo văn (mở rộng)", key="audit_plagiarism"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang đối chiếu n-gram và phân tích diễn đạt..."):
                    overlaps = internal_overlap_audit(audit_text, top_k=5) # Sửa lỗi xác định overlaps
                    response = plagiarism_style_review(audit_text)
                
                with ket_qua_audit_container:
                    st.markdown("### 📄 Kết quả kiểm tra Nguy cơ đạo văn")
                    if overlaps:
                        max_sim = max(o["similarity"] for o in overlaps)
                        if max_sim >= 0.3:
                            st.error(f"Tỷ lệ trùng n-gram cao nhất: {max_sim*100:.1f}% — cần xem lại diễn đạt.")
                        elif max_sim >= 0.1:
                            st.warning(f"Tỷ lệ trùng n-gram cao nhất: {max_sim*100:.1f}% — nên kiểm tra thêm.")
                        else:
                            st.info(f"Tỷ lệ trùng n-gram cao nhất: {max_sim*100:.1f}% — khá thấp.")
                    else:
                        st.info("Không phát hiện trùng cụm từ dài.")
                    if response: st.markdown(response)
 
    with b3:
        if st.button("🤖 Chỉ báo nguy cơ văn bản do AI viết", key="audit_ai_style"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang phân tích phong cách văn bản..."):
                    style_analysis = heuristic_ai_style_score(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🤖 Chỉ báo nguy cơ văn bản do AI viết")
                    if style_analysis: st.markdown(style_analysis)

    st.write("---")
    st.subheader("Phản biện logic bằng AI")
    logic_request = st.text_area("Mô tả vấn đề hoặc dán đoạn văn", height=180, key="logic_request")
    if st.button("⚖️ Phản biện logic", key="logic_review"):
        if not logic_request.strip():
            st.warning("Nhập nội dung cần phản biện.")
        else:
            with st.spinner("AI đang xử lý..."):
                prompt = f"{BASE_SYSTEM_RULES}\nĐóng vai phản biện luận văn CKI Dược lâm sàng, kiểm tra tính logic của:\n{logic_request}"
                response = call_gemini(prompt)
                if response: st.markdown(response)


# ------------------------------------------------------------
# TAB 6 – NGUỒN, CITATION & QUẢN LÝ DỰ ÁN (CHECKPOINT)
# ------------------------------------------------------------
with tabs[5]:
    st.header("⚙️ Nguồn, Citation & Quản lý Dự án")

    st.write(f"**Gemini Model:** `{DEFAULT_MODEL}`")
    st.write(f"**Embedding Model:** `{DEFAULT_EMBEDDING}`")
    st.write(f"**Citation Registry hiện có:** {len(st.session_state['citation_registry'])} nguồn")

    st.write("---")
    st.subheader("💾 Quản lý Checkpoint & Dự án Luận văn")
    st.info("Lưu lại toàn bộ trạng thái nghiên cứu xuống hệ thống để tránh mất dữ liệu khi restart app.")
    
    col_save1, col_save2 = st.columns([2, 1])
    with col_save1:
        proj_name_input = st.text_input("Tên định danh đề tài / dự án:", value="THA_BVTP_2026")
    with col_save2:
        st.write("")
        st.write("")
        if st.button("💾 Lưu Checkpoint Dự án", type="primary"):
            success, msg = save_project(proj_name_input)
            if success: st.success(msg)
            else: st.error(msg)

    st.write("📂 **Danh sách dự án đã lưu trên máy chủ:**")
    existing_projects = list_projects()
    if existing_projects:
        selected_proj = st.selectbox("Chọn dự án để khôi phục:", existing_projects)
        if st.button("📂 Tải lại Checkpoint này"):
            success, msg = load_project(selected_proj)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    else:
        st.caption("Chưa có dự án nào được lưu.")

    st.write("---")
    st.subheader("Danh mục tham khảo hiện tại")
    registry = st.session_state["citation_registry"]
    if registry:
        st.code(citation_bibliography(), language="text")
    else:
        st.info("Citation registry chưa có dữ liệu.")

    st.write("---")
    if st.button("📄 Xuất bản nháp hiện tại ra Word", key="export_word"):
        if not st.session_state["last_generated"]:
            st.warning("Chưa có bản nháp.")
        else:
            docx_data = create_word_document(
                title="Bản nháp hỗ trợ nghiên cứu – Dược lâm sàng",
                body=st.session_state["last_generated"],
                bibliography=citation_bibliography(),
            )
            st.download_button(
                "📥 Tải Word", data=docx_data,
                file_name="Ban_nhap_NCKH_Duoc_Lam_Sang.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
