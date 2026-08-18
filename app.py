# app.py
# ============================================================
# HỖ TRỢ NGHIÊN CỨU KHOA HỌC – EVIDENCE-BASED RAG
# Bản tối ưu cho luận văn Chuyên khoa cấp I – Dược lâm sàng
# (Đã rà soát & sửa lỗi logic, giữ nguyên toàn bộ chức năng gốc)
# ============================================================

import io
import os
import re
import time
import math
import uuid
import json
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
    crosstab_test, compare_two_groups, binary_logistic_regression,
    create_clinical_groups, generate_baseline_table
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
    "jmpm.vn",
    "huejmp.vn",
    "tcydls108.benhvien108.vn",
    "tapchiyhcd.vn",
    "thaibinhjmp.vn",
    "hup.edu.vn",
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
    """Khởi tạo toàn bộ Session State, tách biệt dữ liệu sống và UI."""
    if "ui_version" not in st.session_state:
        st.session_state["ui_version"] = 0

    defaults = {
        # Evidence DB
        "documents": {},
        "chunks": [],
        "embeddings": None,
        "bm25": None,
        "citation_registry": {},

        # Theo dõi & Audit
        "Audit_log": [],
        "last_generated": "",
        "last_evidence": [],
        "current_references": [],

        # Nguồn API/Cào dữ liệu
        "vn_journal_domains": list(DEFAULT_VN_JOURNAL_DOMAINS),
        "t3_pm_data": [],
        "t3_vn_data": [],
        "t3_en_keyword": "",
        "t3_query": "",

        # Thống kê & Tuyển chọn
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
# 5. GEMINI CLIENT (HỖ TRỢ XOAY VÒNG NHIỀU API KEY)
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

def get_api_keys() -> List[str]:
    keys_list = []
    try:
        # Tự động quét mọi biến trong st.secrets chứa "GEMINI_API" (ví dụ: GEMINI_API_KEY, GEMINI_API_KEYS, GEMINI_API_KEY_2,...)
        for key_name, value in st.secrets.items():
            if "GEMINI_API" in key_name and isinstance(value, str):
                cleaned_val = value.replace("\n", ",").replace("\r", ",")
                for k in cleaned_val.split(","):
                    if k.strip():
                        keys_list.append(k.strip())
    except Exception:
        pass
    
    # Dự phòng tìm trong biến môi trường hệ thống nếu không đọc được secrets
    if not keys_list:
        for env_key in ["GEMINI_API_KEYS", "GEMINI_API_KEY"]:
            env_val = os.getenv(env_key, "")
            if env_val:
                cleaned_env = env_val.replace("\n", ",").replace("\r", ",")
                for k in cleaned_env.split(","):
                    if k.strip():
                        keys_list.append(k.strip())

    # Loại bỏ các key trùng lặp nhưng giữ nguyên thứ tự sắp xếp
    return list(dict.fromkeys(keys_list))

def call_gemini(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 5,
) -> Optional[str]:
    api_keys = get_api_keys()
    if not api_keys:
        st.error("Chưa có GEMINI_API_KEYS. Hãy thêm vào Streamlit Secrets.")
        return None
        
    model_name = model or DEFAULT_MODEL
    
    # Biến nhớ xem hệ thống đang dùng tới key số mấy
    if "current_key_idx" not in st.session_state:
        st.session_state["current_key_idx"] = 0

    for attempt in range(max_retries):
        # Lấy key hiện tại dựa theo index
        current_idx = st.session_state["current_key_idx"] % len(api_keys)
        current_key = api_keys[current_idx]
        
        client = get_gemini_client(current_key)

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
            # Bắt lỗi quá tải (429) hoặc hết Quota
            if any(code in error_msg for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "quota"]):
                if len(api_keys) > 1:
                    # NẾU CÓ NHIỀU KEY: Nhảy sang key tiếp theo ngay lập tức
                    st.session_state["current_key_idx"] += 1
                    next_idx = st.session_state["current_key_idx"] % len(api_keys)
                    status = st.warning(f"🔄 Key số {current_idx + 1} đang bận/hết hạn mức. Hệ thống tự động chuyển sang Key số {next_idx + 1}...")
                    time.sleep(1.5)
                    status.empty()
                    continue # Bỏ qua việc chờ, lặp lại vòng while ngay với key mới
                else:
                    # NẾU CÓ 1 KEY: Chờ 15s như cũ
                    if attempt < max_retries - 1:
                        wait_time = 15  
                        status = st.warning(f"⏳ Trạm máy chủ Google đang quá tải. Đợi {wait_time} giây rồi thử lại...")
                        time.sleep(wait_time)
                        status.empty()  
                    else:
                        st.error("❌ Máy chủ Google Gemini hiện đang quá bận. Vui lòng đợi 1-2 phút rồi bấm thử lại!")
                        return None
            else:
                # Lỗi khác không phải quá tải
                if attempt == max_retries - 1:
                    st.error(f"Lỗi Gemini: {error_msg}")
                    return None
                time.sleep(3)

    return None
# ============================================================
# HÀM MỚI: AI TỰ ĐỘNG ĐỌC METADATA TỪ TRANG ĐẦU CỦA PDF
# ============================================================
def extract_metadata_from_text_ai(text: str) -> dict:
    """Hàm nhờ AI đọc 4000 ký tự đầu tiên của file PDF để bóc tách thông tin tác giả, năm..."""
    prompt = f"""Bạn là một chuyên gia thư viện y khoa.
Nhiệm vụ: Đọc đoạn văn bản (trích từ trang đầu của bài báo/tài liệu PDF) và tìm các thông tin thư mục.
TRẢ VỀ DUY NHẤT MỘT CHUỖI JSON HỢP LỆ, KHÔNG GIẢI THÍCH GÌ THÊM:
{{
    "authors": "Tên các tác giả",
    "title": "Tên bài báo / tài liệu",
    "year": "Năm xuất bản",
    "journal": "Tên tạp chí hoặc nhà xuất bản",
    "doi": "Mã DOI (nếu có)",
    "pmid": "Mã PMID (nếu có)"
}}
Nếu không tìm thấy thông tin nào, hãy để giá trị là chuỗi rỗng "".
ĐOẠN VĂN BẢN:
{text[:4000]}
"""
    res = call_gemini(prompt, model=MODEL_LITE, temperature=0.1)
    if not res: 
        return {}
    try:
        # Xóa các ký tự markdown an toàn không dùng replace
        cleaned = res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        return json.loads(cleaned.strip())
    except Exception:
        return {}

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
# 8. TRA CỨU TỪ KHÓA (ĐÃ TỐI ƯU HÓA CHO PUBMED)
# ============================================================
def translate_query_to_mesh(vietnamese_query: str) -> str:
    prompt = f"""Bạn là một chuyên gia tra cứu tài liệu y khoa.
Nhiệm vụ: Chuyển đổi tên đề tài tiếng Việt sau thành chuỗi từ khóa tiếng Anh hiệu quả nhất để tra cứu trên PubMed.
NGUYÊN TẮC BẮT BUỘC:
1. Trích xuất các danh từ chính (Ví dụ: Vancomycin, drug utilization, pharmacokinetics).
2. Dùng toán tử AND, OR để kết nối.
3. TUYỆT ĐỐI KHÔNG dùng cấu trúc MeSH chứa dấu gạch chéo.
4. Giữ chuỗi tìm kiếm ngắn gọn, từ 2 đến 4 cụm từ khóa.
Từ gốc tiếng Việt: "{vietnamese_query}"
Chỉ trả về chuỗi tiếng Anh, không giải thích, không dùng markdown."""
    text = call_gemini(prompt, model=MODEL_LITE)
    if text:
        return text.strip().strip('"').strip("'").replace("\n", " ")
    return vietnamese_query

def extract_vn_keywords(vietnamese_query: str) -> str:
    prompt = f"""Bạn là chuyên gia tra cứu tài liệu.
Nhiệm vụ: Rút gọn tên đề tài sau thành 1-2 từ khóa quan trọng nhất bằng tiếng Việt.
KHÔNG dùng câu dài. CHỈ giữ lại danh từ cốt lõi (hoạt chất, tên bệnh).
Đề tài: "{vietnamese_query}"
Chỉ trả về từ khóa rút gọn, không giải thích."""
    text = call_gemini(prompt, model=MODEL_LITE)
    if text:
        return text.strip().strip('"').strip("'").replace("\n", " ")
    return vietnamese_query

# ============================================================
# 10. INDEX / VECTOR RETRIEVAL (HYBRID: EMBEDDING + BM25)
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
        f'{"" if summary["index_ready"] else " &nbsp;—&nbsp; ⚠️ chưa dựng xong index, thử tải lại"}</div>'
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
# 11. CITATION ENGINE (CẤU TRÚC MỚI CHUẨN VANCOUVER)
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
        
        # Mẹo: Dùng "or" để nếu giá trị là rỗng ("") thì lấy chuỗi dự phòng
        authors = meta.get('authors') or "Tác giả chưa xác định"
        title = meta.get('title') or meta.get('file_name') or "Tài liệu chưa xác định"
        journal = meta.get('journal') or "Tài liệu lưu trữ"
        year = meta.get('year') or "Năm chưa rõ"
        
        citation = f"[{ref['vancouver_index']}] {authors}. {title}. {journal}. {year}."
        
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
1. Tài liệu được cung cấp (PDF, tóm tắt PubMed, đoạn trích tạp chí Việt Nam) là nguồn bằng chứng ưu tiên duy nhất.
2. Không tự tạo số liệu, p-value, OR, RR, HR, CI95%, tỷ lệ %, liều dùng hoặc cỡ mẫu nếu không có trong bằng chứng.
3. Không tự tạo tên tác giả, năm, tên bài báo, DOI, PMID.
4. Nếu context không đủ bằng chứng, phải nói rõ: "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận."
5. Mọi khẳng định dựa trên tài liệu phải chèn MÃ ĐỊNH DANH của tài liệu đó ngay sau câu. Ví dụ: "Tỷ lệ này là 12% [REF-001]."
6. TUYỆT ĐỐI KHÔNG tự tạo [1], [2], [3] và không dùng citation dạng tác giả-năm.
7. Phân biệt rõ: FACT (dữ kiện) – INTERPRETATION (diễn giải) – INFERENCE (suy luận).
8. Dùng chính xác thuật ngữ chuyên ngành Dược lâm sàng, văn phong khô khan.
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

    prompt = f"""
{BASE_SYSTEM_RULES}

NHIỆM VỤ:
{task}

BẰNG CHỨNG ĐƯỢC PHÉP SỬ DỤNG:
{evidence_context}

YÊU CẦU:
- LƯU Ý: KHÔNG ĐƯỢC tự đánh số [1], [2]. PHẢI dùng nguyên vẹn mã [REF-...] từ tài liệu.
- Chỉ sử dụng thông tin có thể truy về các tài liệu ở trên.
"""

    output = call_gemini(prompt)
    if output is None:
        return None, evidence, []

    final_text, references, invalid_tags = engine.process_vancouver_citations(output)

    if invalid_tags:
        final_text += (
            f"\n\n> ⚠️ CẢNH BÁO Audit: Phát hiện AI tự tạo mã trích dẫn không có "
            f"trong dữ liệu truy xuất: {', '.join(invalid_tags)}. Đoạn này cần Audit kỹ."
        )

    st.session_state["last_generated"] = final_text
    st.session_state["last_evidence"] = evidence
    st.session_state["current_references"] = references

    return final_text, evidence, invalid_tags

# ============================================================
# 17. Audit NHẤT QUÁN SỐ LIỆU + TRÙNG LẶP NỘI BỘ
# ============================================================
def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*%)?"
    return re.findall(pattern, text)

def parse_number(num_str: str) -> Optional[float]:
    clean_str = num_str.replace(",", ".").replace(" ", "").replace("%", "")
    try:
        return float(clean_str)
    except ValueError:
        return None

def _strip_citation_markers(text: str) -> str:
    """
    Loại bỏ các mã trích dẫn dạng [1], [2]... (đã được CitationEngine đánh số)
    và các mã thô REF-xxx (trường hợp văn bản chưa qua xử lý citation) trước khi
    tách số. Nếu không làm bước này, số thứ tự trích dẫn (vd. "[1]") sẽ bị hiểu
    nhầm thành số liệu và bị báo sai là "số liệu lạ" trong Audit.
    """
    if not text:
        return text
    cleaned = re.sub(r"\[\s*\d+\s*\]", " ", text)
    cleaned = re.sub(r"REF-[A-Za-z0-9\-]+", " ", cleaned)
    return cleaned

def compare_numbers_advanced(source_text: str, generated_text: str) -> Dict[str, Any]:
    generated_text_clean = _strip_citation_markers(generated_text)

    source_nums = extract_numeric_tokens(source_text)
    generated_nums = extract_numeric_tokens(generated_text_clean)

    source_normalized = set(x.replace(",", ".").replace(" ", "") for x in source_nums)
    generated_normalized = set(x.replace(",", ".").replace(" ", "") for x in generated_nums)
    source_floats = set(filter(None, [parse_number(x) for x in source_normalized]))

    exact_matches = []
    derived_matches = []
    warnings = []

    for gen_num in generated_normalized:
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

                if is_derived:
                    derived_matches.append(gen_num)
                else:
                    warnings.append(gen_num)
            else:
                warnings.append(gen_num)

    return {
        "exact_matches": sorted(exact_matches),
        "derived_matches": sorted(derived_matches),
        "warnings": sorted(warnings),
        "source_raw": sorted(source_normalized)
    }

def Audit_generated_text(text: str) -> Dict[str, Any]:
    relevant_evidence = retrieve_evidence(text, k=6)
    source_text = "\n".join(e["text"] for e in relevant_evidence)
    audit_result = compare_numbers_advanced(source_text, text)
    return {
        "evidence_used": relevant_evidence,
        **audit_result
    }

def normalize_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.,-]", "", text)
    return text.strip()

def ngram_set(text: str, n: int = 8) -> set:
    words = normalize_for_similarity(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

def internal_overlap_Audit(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
    """
    Chuyển nội dung Markdown (heading, bullet, bảng, đoạn văn) vào Document.

    LƯU Ý QUAN TRỌNG (đã sửa lỗi): trước đây cờ `_last_table_open` chỉ bị tắt khi
    gặp một dòng VĂN BẢN THƯỜNG, nên nếu 2 bảng markdown liên tiếp chỉ cách nhau
    bởi một dòng TIÊU ĐỀ (### ...) hoặc dòng TRỐNG thì bảng thứ 2 sẽ bị nối nhầm
    (thêm hàng) vào bảng thứ 1 thay vì tạo bảng mới — gây sai lệch dữ liệu khi
    xuất nhiều bảng thống kê ra cùng một file Word (mục "Tải TẤT CẢ bảng ra Word").
    Nay bất kỳ dòng nào KHÔNG PHẢI là dòng bảng (kể cả heading, bullet, dòng
    trống) đều sẽ đóng bảng hiện tại lại trước khi xử lý tiếp.
    """
    lines = body.split("\n")
    buffer_paragraph = []

    def flush_paragraph():
        if buffer_paragraph:
            doc.add_paragraph(" ".join(buffer_paragraph).strip())
            buffer_paragraph.clear()

    for line in lines:
        stripped = line.strip()
        is_table_row = stripped.startswith("|")

        # Đóng bảng hiện tại nếu dòng này không phải là dòng bảng markdown.
        if not is_table_row:
            doc._last_table_open = False

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
        elif is_table_row:
            if set(stripped.replace("|", "").replace(" ", "").replace(":", "")) <= {"-"}:
                continue
            flush_paragraph()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            table = doc.tables[-1] if (doc.tables and getattr(doc, "_last_table_open", False)) else None

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
# 18.5. CÁC HÀM XỬ LÝ CHO TAB Audit
# ============================================================
def spelling_and_terminology_check(text: str) -> Optional[str]:
    if not text.strip():
        return None
    prompt = f"""{BASE_SYSTEM_RULES}
Bạn là một biên tập viên y khoa khó tính chuyên ngành Dược lâm sàng.
Hãy rà soát đoạn văn bản sau để tìm ra các lỗi chính tả, đánh máy, sai thuật ngữ chuyên ngành.
ĐOẠN VĂN GỐC:
{text}
Chỉ trình bày những lỗi tìm thấy và đề xuất cách sửa. Nếu không có lỗi, hãy báo "✅ Không tìm thấy lỗi chính tả/thuật ngữ đáng kể"."""
    return call_gemini(prompt, model=MODEL_LITE)

def plagiarism_style_review(text: str) -> Optional[str]:
    if not text.strip():
        return None
    prompt = f"""{BASE_SYSTEM_RULES}
Hãy đóng vai hội đồng phản biện luận văn CKI Dược lâm sàng.
Phân tích đoạn văn sau để đánh giá độ logic mạch lạc, cảnh báo câu cấu trúc lặp lại (nguy cơ đạo văn), đề xuất cách nâng cấp văn phong.
ĐOẠN VĂN GỐC:
{text}"""
    return call_gemini(prompt)

def heuristic_ai_style_score(text: str) -> Optional[str]:
    if not text.strip():
        return None
    prompt = f"""{BASE_SYSTEM_RULES}
Hãy phân tích đoạn văn sau và soi khắt khe các dấu hiệu văn bản do AI viết: từ nối rập khuôn (Tóm lại, Nhìn chung...), cấu trúc câu máy móc, tính từ hoa mỹ.
ĐOẠN VĂN GỐC:
{text}
Trình bày gạch đầu dòng các dấu hiệu tìm thấy và hướng dẫn sửa lại cho tự nhiên."""
    return call_gemini(prompt, model=MODEL_LITE)

# ============================================================
# 19. GIAO DIỆN CHÍNH
# ============================================================
def main():
    init_state()

    st.title("🔬 HỖ TRỢ NGHIÊN CỨU KHOA HỌC")
    st.caption(
        "Evidence-Based RAG • Tra cứu TLTK (PubMed + Tạp chí VN) • "
        "Citation Registry • Statistical Engine • Audit"
    )

    tabs = st.tabs([
        "📚 1. Tài liệu (PDF)",
        "🔍 2. Tra cứu TLTK",
        "✍️ 3. Viết luận văn",
        "📊 4. Phân tích số liệu",
        "🔎 5. Kiểm tra luận văn",
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
                    for err in errors:
                        st.error(err)

        with col2:
            if st.button("🗑️ Xóa toàn bộ ngân hàng tài liệu"):
                st.session_state["documents"] = {}
                st.session_state["chunks"] = []
                st.session_state["embeddings"] = None
                st.session_state["bm25"] = None
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
            placeholder="Ví dụ: tỷ lệ bệnh nhân đạt huyết áp mục tiêu...",
            key=ui_key("evidence_query")
        )
        top_k = st.slider("Số đoạn bằng chứng", 3, MAX_TOP_K, DEFAULT_TOP_K, key=ui_key("top_k_tab1"))

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
                            f"**{ev['chunk_id']}** _( {meta.get('origin', '')} )_\n"
                            f"Nguồn: {ev['file_name']} — Trang/mục: {ev['page'] or ev.get('section', '')}\n"
                            f"Điểm tương đồng: {ev['score']:.4f}\n\n> {ev['text']}"
                        )

    # ------------------------------------------------------------
    # TAB 2 – Tra cứu TLTK
    # ------------------------------------------------------------
    with tabs[1]:
        st.header("🔍 Tra cứu TLTK: PubMed (Quốc tế) + Tạp chí Y học Việt Nam")
        st.info("Nhập tên đề tài bằng tiếng Việt. Hệ thống tự dịch sang từ khoá MeSH để tìm trên PubMed, đồng thời tìm bài báo tiếng Việt liên quan.")
        render_evidence_database_status()

        col_search, col_btn = st.columns([4, 1])
        with col_search:
            t3_query = st.text_input(
                "Tên đề tài nghiên cứu (tiếng Việt):",
                key=ui_key("t3_query_input"),
            )
        with col_btn:
            max_res = st.number_input("Số bài/nguồn", min_value=2, max_value=10, value=5, key=ui_key("t3_max_res"))

        if st.button("🚀 Tra cứu song song 2 nguồn", type="primary", key="t3_btn_search"):
            if not t3_query.strip():
                st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
            else:
                st.session_state["t3_query"] = t3_query

                with st.spinner("Đang dịch & chuẩn hoá từ khoá sang MeSH (PubMed)..."):
                    en_query = translate_query_to_mesh(t3_query)
                    st.session_state["t3_en_keyword"] = en_query

                with st.spinner("Đang tìm & tải Abstract từ PubMed..."):
                    st.session_state["t3_pm_data"] = search_pubmed(en_query, max_res)

                with st.spinner("Đang rút gọn từ khóa & tìm trên tạp chí Y học Việt Nam..."):
                    st.session_state["t3_vn_keyword"] = t3_query 
                    vn_results, vn_err = search_vn_journals(t3_query, max_res)
                    st.session_state["t3_vn_data"] = vn_results
                    if vn_err:
                        st.warning(vn_err)

        if st.session_state.get("t3_pm_data") or st.session_state.get("t3_vn_data"):
            st.write("---")
            col_vn, col_pm = st.columns(2)

            with col_vn:
                st.markdown("### 🇻🇳 Tạp chí Y học Việt Nam")
                if st.session_state.get("t3_vn_keyword"):
                    st.success(f"🔑 Từ khoá đã rút gọn: **{st.session_state['t3_vn_keyword']}**")

                if not st.session_state.get("t3_vn_data"):
                    st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
                else:
                    for i, art in enumerate(st.session_state["t3_vn_data"]):
                        with st.container(border=True):
                            st.markdown(f"**[{art['title']}]({art['link']})**" if art.get("link") else f"**{art['title']}**")
                            st.caption(art["source"])
                            st.write(art["snippet"])
                            if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"vn_ingest_{i}")):
                                added = ingest_vn_article(art)
                                if added:
                                    rebuild_index()
                                    st.success("Đã nạp. Nhớ Audit bản gốc trước khi dùng số liệu chi tiết.")
                                else:
                                    st.info("Nguồn này đã có trong Evidence Database.")

            with col_pm:
                st.markdown("### 🌍 PubMed (Quốc tế)")
                if st.session_state.get("t3_en_keyword"):
                    st.success(f"🔑 Từ khoá MeSH: **{st.session_state['t3_en_keyword']}**")
                if not st.session_state.get("t3_pm_data"):
                    st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
                else:
                    for i, art in enumerate(st.session_state["t3_pm_data"]):
                        with st.container(border=True):
                            st.markdown(f"**[{art['title']}]({art['url']})**")
                            st.caption(f"✍️ {art['authors']} ({art['year']}) — {art['journal']}")
                            with st.expander("Xem tóm tắt (Abstract)"):
                                st.write(art["abstract"])
                            if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"pm_ingest_{i}")):
                                added = ingest_pubmed_article(art)
                                if added:
                                    rebuild_index()
                                    st.success("Đã nạp vào Evidence Database.")
                                else:
                                    st.info("Nguồn này đã có trong Evidence Database.")

            st.write("---")
            if st.button("➕ Nạp TẤT CẢ kết quả ở trên vào Evidence Database", key="t3_ingest_all"):
                count = 0
                for art in st.session_state.get("t3_pm_data", []):
                    if ingest_pubmed_article(art):
                        count += 1
                for art in st.session_state.get("t3_vn_data", []):
                    if ingest_vn_article(art):
                        count += 1
                if count:
                    rebuild_index()
                st.success(f"Đã nạp {count} nguồn mới vào Evidence Database.")

    # ------------------------------------------------------------
    # TAB 3 – VIẾT LUẬN VĂN
    # ------------------------------------------------------------
    with tabs[2]:
        st.header("✍️ Viết luận văn dựa trên bằng chứng")
        st.warning("Đây là công cụ tạo bản nháp. Mọi citation và số liệu phải được Audit lại (đối chiếu bản gốc) trước khi đưa vào luận văn chính thức.")
        render_evidence_database_status("dùng cho các nút viết nhanh bên dưới")

        # ------------------------------------------------------------
        # KHU VỰC NHẬP DỮ LIỆU NGHIÊN CỨU & NHẬN XÉT BẢNG
        # ------------------------------------------------------------
        st.markdown("### 📊 Dữ liệu nghiên cứu của riêng anh")
        
        # MẸO XỬ LÝ LỖI STREAMLIT: Nhồi dữ liệu AI vào state TRƯỚC khi vẽ widget
        if "ai_pending_remark" in st.session_state:
            st.session_state[ui_key("my_table_remarks")] = st.session_state["ai_pending_remark"]
            del st.session_state["ai_pending_remark"]

        col_data_1, col_data_2 = st.columns(2)
        
        with col_data_1:
            my_research_data = st.text_area(
                "1. Số liệu bảng (Dán bảng Excel/Markdown vào đây):",
                placeholder="VD: Bảng phân bố thời gian phẫu thuật...",
                height=180,
                key=ui_key("my_research_data")
            )
            
        with col_data_2:
            my_table_remarks = st.text_area(
                "2. Nhận xét bảng (Chỉ diễn giải số liệu, KHÔNG bàn luận):",
                placeholder="VD: Thời gian phẫu thuật trung vị là 45 phút... Đa số hoàn thành trong dưới 60 phút (90%).\n\n(Anh có thể gõ tay, hoặc bấm nút 'AI Viết Nhận Xét Bảng' ở dưới để AI tự sinh ra).",
                height=180,
                key=ui_key("my_table_remarks")
            )

        if st.button("✍️ AI Viết Nhận Xét Bảng (Tự động điền vào ô 2)"):
            if not my_research_data.strip():
                st.warning("⚠️ Anh cần dán bảng số liệu vào ô số 1 trước để AI có dữ liệu đọc!")
            else:
                with st.spinner("AI đang phân tích bảng và soạn nhận xét..."):
                    task = """Bạn là biên tập viên y khoa. Dựa vào bảng số liệu dưới đây, hãy viết phần 'Nhận xét' chuẩn văn phong luận văn. 
YÊU CẦU TỐI THƯỢNG: 
- Ngắn gọn, logic, khoa học. 
- CHỈ diễn giải các số liệu nổi bật (cao nhất, thấp nhất, trung vị, tỷ lệ %). 
- TUYỆT ĐỐI KHÔNG giải thích nguyên nhân, KHÔNG so sánh với y văn, KHÔNG bàn luận.
- BẮT BUỘC viết thành MỘT ĐOẠN VĂN LIỀN MẠCH duy nhất (Không dùng gạch đầu dòng, không xuống dòng).
- KHÔNG dùng ký tự in đậm (**), KHÔNG dùng ký tự ($), KHÔNG tự chèn tên bảng vào cuối câu."""
                    prompt = f"{BASE_SYSTEM_RULES}\nNHIỆM VỤ:\n{task}\n\nBẢNG SỐ LIỆU:\n{my_research_data}"
                    
                    generated_remark = call_gemini(prompt)
                    
                    if generated_remark:
                        # Lưu vào biến tạm rồi ép tải lại giao diện
                        st.session_state["ai_pending_remark"] = generated_remark
                        st.rerun()

        st.write("---")

        citation_rules = """QUY TẮC TRÍCH DẪN & HÀN LÂM BẮT BUỘC:
Chỉ dùng SOURCE_TAG thật để hệ thống tự chuyển thành [n].
Các số trích dẫn sẽ theo đúng thứ tự xuất hiện trong bài (do hệ thống cấp).
Không dùng kiểu [Tên tác giả, Năm].
Không dùng Heading 1 (#) hoặc Heading 2 (##) trong nội dung, chỉ dùng Heading 3 (###) cho tiêu đề mục."""

        ket_qua_container = st.container()

        def run_quick_task(task_label: str, query: str, task_prompt: str, k: int):
            with st.spinner(f"AI đang soạn: {task_label}..."):
                output, evidence, invalid = generate_evidence_based(task_prompt, query, k=k)
                if output:
                    with ket_qua_container:
                        st.write("---")
                        st.subheader(task_label)
                        st.markdown(output)

                        st.markdown("### 🔎 Dấu vết bằng chứng (Evidence Trace)")
                        st.caption("Bấm vào từng trích dẫn để xem chính xác đoạn văn bản gốc mà AI đã dùng làm căn cứ.")

                        current_refs = st.session_state.get("current_references", [])
                        if current_refs:
                            for ref in current_refs:
                                v_index = ref['vancouver_index']
                                ref_id = ref['ref_id']
                                source_id = ref_id.replace("REF-", "") if ref_id.startswith("REF-") else ref_id

                                related_chunks = [ev for ev in evidence if ev['source_id'] == source_id]
                                title = ref['metadata'].get('title', 'Tài liệu chưa có tiêu đề')
                                file_name = ref['metadata'].get('file_name', 'N/A')

                                with st.expander(f"[{v_index}] ↳ {title[:85]}..."):
                                    st.write(f"**Tệp gốc:** `{file_name}`")
                                    if ref['metadata'].get('doi'):
                                        st.write(f"**DOI:** {ref['metadata']['doi']}")

                                    for chunk in related_chunks:
                                        st.markdown(
                                            f"- **Trang/Mục:** `{chunk.get('page', 'N/A')}` | "
                                            f"**Độ khớp:** `{chunk.get('score', 0):.4f}` | "
                                            f"**Mã đoạn:** `{chunk.get('chunk_id', 'N/A')}`"
                                        )
                                        st.info(f"_{chunk.get('text', '')}_")
                        else:
                            st.info("Đoạn văn này không sử dụng trích dẫn nào từ Evidence Database.")

                        bib = citation_bibliography()
                        with st.expander("📖 Danh mục Tài liệu tham khảo (Của bản nháp này)"):
                            st.code(bib if bib else "Chưa có citation registry.", language="text")

                        Audit = Audit_generated_text(output)
                        colA, colB = st.columns(2)
                        with colA:
                            if invalid:
                                st.error(f"Phát hiện citation ảo: {', '.join(invalid)}")
                            else:
                                st.success("Không phát hiện citation ảo.")
                        with colB:
                            if Audit.get("warnings"):
                                st.warning(f"Số liệu lạ (Cần Audit lại): {', '.join(Audit['warnings'])}")
                            else:
                                st.success("Không phát hiện số liệu lạ ngoài bằng chứng.")

                        st.session_state["Audit_log"].append({
                            "type": task_label, "invalid_citation": invalid, "Audit": Audit,
                        })

        st.subheader("📝 Lệnh viết nhanh")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            btn_dat_van_de = st.button("Đặt vấn đề")
        with c2:
            btn_tong_quan = st.button("Tổng quan tài liệu")
        with c3:
            btn_phuong_phap = st.button("Phương pháp NC")
        with c4:
            btn_ban_luan = st.button("Bàn luận KQNC và So sánh")
        with c5:
            btn_tltk = st.button("Trích dẫn TLTK")

        st.write("---")
        st.subheader("Lệnh tùy chỉnh")
        custom_prompt = st.text_area("Nhập câu lệnh khác:", key=ui_key("custom_prompt_tab3"))
        k_custom = st.slider("Số nguồn bằng chứng truy xuất", 3, MAX_TOP_K, DEFAULT_TOP_K, key=ui_key("top_k_tab3"))
        btn_custom = st.button("▶️ Chạy lệnh tùy chỉnh")

        if btn_dat_van_de:
            query = "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, dịch tễ học, gánh nặng bệnh tật liên quan sử dụng thuốc"
            task = f"Viết phần 'Đặt vấn đề' cho luận văn CKI Dược lâm sàng.\nYÊU CẦU BẮT BUỘC VỀ HÌNH THỨC:\nViết thành MỘT MẠCH VĂN LIỀN MẠCH, khoảng 400 từ, gồm 3-4 đoạn văn...\n{citation_rules}"
            run_quick_task("Đặt vấn đề", query, task, k=6)

        if btn_tong_quan:
            query = "Tổng quan y văn, các nghiên cứu liên quan, cơ chế dược lý, kết quả chính, khuyến cáo điều trị"
            task = f"Viết phần 'Tổng quan tài liệu' chuyên sâu...\n{citation_rules}"
            run_quick_task("Tổng quan tài liệu", query, task, k=8)

        if btn_phuong_phap:
            query = "Đối tượng nghiên cứu, tiêu chuẩn chọn loại, thiết kế nghiên cứu, cỡ mẫu, biến số nghiên cứu"
            task = f"Viết 'Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU'...\n{citation_rules}"
            run_quick_task("Phương pháp nghiên cứu", query, task, k=5)

        if btn_ban_luan:
            if not my_research_data.strip() and not my_table_remarks.strip():
                st.warning("⚠️ Cần dán bảng số liệu (ô 1) và nhận xét (ô 2) của anh vào phía trên trước!")
            else:
                context = f"SỐ LIỆU BẢNG:\n{my_research_data}\n\nNHẬN XÉT DIỄN GIẢI:\n{my_table_remarks}"
                task = f"""DỮ LIỆU NGHIÊN CỨU THỰC TẾ CỦA TÔI:
{context}

YÊU CẦU TRÌNH BÀY (BÀN LUẬN & SO SÁNH):
Từ phần nhận xét diễn giải ở trên, hãy viết một phần BÀN LUẬN TOÀN DIỆN. Hãy thực hiện mượt mà 2 nhiệm vụ sau trong cùng một mạch văn:
1. Giải thích sâu sắc nguyên nhân hoặc cơ sở lâm sàng có thể dẫn đến kết quả này.
2. Đối chiếu, so sánh trực tiếp kết quả của tôi với các tác giả và nghiên cứu khác trong y văn.

CÁC RÀNG BUỘC BẮT BUỘC:
- BẮT BUỘC viết thành các đoạn văn xuôi y khoa liền mạch, tự nhiên (tuyệt đối không dùng gạch đầu dòng để phân ý, không xuống dòng ngắt vụn).
- TUYỆT ĐỐI KHÔNG sử dụng các nhãn phân chia máy móc như "Dữ kiện (FACT):", "Diễn giải (INTERPRETATION):", hay "Suy luận (INFERENCE):".
- Nếu tài liệu không có thông tin để đối chứng, hãy lướt qua hoặc diễn đạt tự nhiên (VD: "Tuy nhiên, y văn hiện tại chưa ghi nhận số liệu tương đương..."). TUYỆT ĐỐI KHÔNG dùng câu rập khuôn "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận...".
- Trích dẫn tài liệu tham khảo phải theo thứ tự xuất hiện hiện của tài liệu tham khảo.

{citation_rules}"""
                run_quick_task("Bàn luận và So sánh toàn diện", context, task, k=8)
                
        if btn_tltk:
            query = "Tài liệu tham khảo, tác giả, năm xuất bản, tạp chí"
            task = f"Chỉ liệt kê các SOURCE_TAG bạn thấy phù hợp là tài liệu tham khảo chính...\n{citation_rules}"
            with st.spinner("AI đang soạn: Trích dẫn TLTK..."):
                output, evidence, invalid = generate_evidence_based(task, query, k=10)
                if output:
                    with ket_qua_container:
                        st.write("---")
                        st.subheader("Danh mục Tài liệu tham khảo (từ Citation Registry)")
                        bib = citation_bibliography()
                        st.code(bib if bib else "Chưa có citation registry.", language="text")

        if btn_custom:
            if not custom_prompt.strip():
                st.warning("Vui lòng nhập yêu cầu!")
            else:
                task = f"{custom_prompt}\n{citation_rules}"
                run_quick_task("Kết quả lệnh tùy chỉnh", custom_prompt, task, k=k_custom)

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
                        for log in clean_logs:
                            st.write(log)

                validation = validate_dataframe(df)
                if validation:
                    for item in validation:
                        st.warning(item)

                with st.expander("Xem dữ liệu sau khi chuẩn hóa"):
                    st.dataframe(df)

                # ==========================================
                # HIỂN THỊ GIỎ KẾT QUẢ VÀ TẢI VỀ WORD
                # ==========================================
                st.write("---")
                st.markdown(f"### 🛒 Giỏ kết quả: **{len(st.session_state.get('result_cart', []))}** bảng đã lưu")
                st.info("💡 Mỗi khi anh bấm các nút thống kê bên dưới, kết quả tự động được nạp vào Giỏ này để lát nữa AI tuyển chọn hoặc xuất ra Word.")

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
                            rows = []
                            for _, row in df_table.iterrows():
                                rows.append("| " + " | ".join(str(x) for x in row.values) + " |")

                            md_content += "\n".join([header, separator] + rows) + "\n\n"

                        docx_data = create_word_document(
                            title="Phụ lục Số liệu Thống kê (Xuất từ Giỏ kết quả)",
                            body=md_content,
                            bibliography=""
                        )

                        st.download_button(
                            label="📥 Tải TẤT CẢ bảng ra file Word",
                            data=docx_data,
                            file_name="Phu_luc_So_lieu_Thong_ke.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    else:
                        st.button("📥 Tải TẤT CẢ bảng ra file Word", disabled=True, use_container_width=True)

                st.write("---")

                # ==========================================
                # BỘ MÁY TUYỂN CHỌN
                # ==========================================
                st.subheader("📋 Bộ máy tuyển chọn & Sắp xếp bảng cho Chương Kết quả")
                with st.expander("🎯 Khai báo Mục tiêu nghiên cứu (Gợi ý: Dùng chính xác tên cột trong Excel làm từ khóa)"):
                    obj_input_1 = st.text_input("Mục tiêu 1", value="ĐẶC ĐIỂM BỆNH NHÂN NGHIÊN CỨU", key=ui_key("obj_1"))
                    obj_input_2 = st.text_input("Mục tiêu 2", value="PHÂN TÍCH THỰC TRẠNG SỬ DỤNG THUỐC", key=ui_key("obj_2"))

                    objectives = [
                        StudyObjective(id="MT1", title=obj_input_1, keywords=["tuổi", "tuoi", "giới", "gioi", "bệnh", "benh", "đặc điểm", "nhân khẩu", "bmi", "SoBHYT", "NgaySinh"]),
                        StudyObjective(id="MT2", title=obj_input_2, keywords=["thuốc", "thuoc", "phù hợp", "phu hop", "liều", "lieu", "chỉ định", "chi dinh", "hoạt chất", "icd", "TenHang"]),
                    ]

                if st.button("🚀 Chạy Table Selection Engine & Lập mạch kể chuyện", type="primary", key="run_engine"):
                    if not st.session_state["result_cart"]:
                        st.error("❌ Giỏ kết quả đang trống! Anh cần cuộn xuống dưới, bấm các nút thống kê để nạp số liệu vào Giỏ trước.")
                    else:
                        engine = TableSelectionEngine(objectives, st.session_state["result_cart"])
                        decisions = engine.run()
                        narrative_plan = NarrativePlanner.build(decisions)

                        st.session_state["selection_decisions"] = decisions
                        st.session_state["narrative_plan"] = narrative_plan
                        st.success("✅ Đã hoàn thành tuyển chọn, lọc trùng và sắp xếp cấu trúc Chương Kết quả!")

                if st.session_state.get("selection_decisions"):
                    st.write("### 📊 Bảng tổng hợp đề xuất cấu trúc Chương 3")
                    display_rows = []
                    for d in st.session_state["selection_decisions"]:
                        display_rows.append({
                            "Thứ tự": d.recommended_order or "Phụ lục",
                            "Mức độ": d.priority.value,
                            "Hình thức": d.presentation.value,
                            "Điểm": d.total_score,
                            "Tiêu đề bảng": d.title,
                            "Lý do đề xuất": d.reason
                        })
                    st.dataframe(pd.DataFrame(display_rows))

                    st.write("### 🖨️ XEM & COPY CÁC BẢNG ĐÃ ĐƯỢC CHỌN (Sẵn sàng đưa vào Word)")
                    for d in st.session_state["selection_decisions"]:
                        if d.result_id in st.session_state["saved_tables"]:
                            st.markdown(f"**Bảng {d.recommended_order or '*'}. {d.title}** *(Xếp loại: {d.priority.value})*")
                            df_table = st.session_state["saved_tables"][d.result_id]
                            html_table = df_table.to_html(index=False, justify='center', border=1)
                            st.markdown(html_table, unsafe_allow_html=True)
                            st.write("<br>", unsafe_allow_html=True)

                    st.write("### 📖 Mạch kể chuyện (Result Story / Narrative Plan)")
                    st.json(st.session_state["narrative_plan"])

                st.write("---")

                # ==========================================
                # 1. THỐNG KÊ MÔ TẢ
                # ==========================================
                st.subheader("1. Thống kê mô tả (biến phân loại)")
                all_cols = df.columns.tolist()
                desc_vars = st.multiselect("Chọn biến phân loại", all_cols, key=ui_key("desc_vars"))

                if st.button("Tính tần số và tỷ lệ (Tự động nạp vào Giỏ)", key="calc_desc"):
                    if not desc_vars:
                        st.warning("Vui lòng chọn ít nhất 1 biến.")
                    else:
                        for var in desc_vars:
                            result = descriptive_table(df, var)
                            if not result.empty:
                                st.markdown(f"**► Biến: {var}**")
                                st.dataframe(result)

                                result_id = f"DESC_{var}"
                                st.session_state["saved_tables"][result_id] = result
                                st.session_state["result_cart"].append(
                                    CandidateResult(
                                        id=result_id, title=f"Đặc điểm phân bố của biến {var}",
                                        result_type="demographic", variables=[var],
                                        scientific_value=3.5, clinical_importance=4.0, discussion_value=3.0
                                    )
                                )
                        st.success(f"✅ Đã nạp {len(desc_vars)} bảng mô tả vào Giỏ!")

                st.write("---")

                # ==========================================
                # 2. BIẾN ĐỊNH LƯỢNG
                # ==========================================
                st.subheader("2. Biến định lượng — Mô tả")
                numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if numeric_candidates:
                    num_vars = st.multiselect("Chọn biến định lượng", numeric_candidates, key=ui_key("num_vars"))
                    if st.button("Tính Mean/SD và Median/IQR (Tự động nạp vào Giỏ)", key="calc_num"):
                        if not num_vars:
                            st.warning("Vui lòng chọn ít nhất 1 biến.")
                        else:
                            for var in num_vars:
                                summary = numeric_summary(df, var)
                                if summary:
                                    st.markdown(f"**► Biến: {var}**")
                                    num_df = pd.DataFrame([{
                                        "N": summary['n'],
                                        "Mean ± SD": f"{summary['mean']:.2f} ± {summary['sd']:.2f}",
                                        "Median (IQR)": f"{summary['median']:.2f} ({summary['q1']:.2f} - {summary['q3']:.2f})",
                                        "Min-Max": f"{summary['min']:.2f} - {summary['max']:.2f}"
                                    }])
                                    st.dataframe(num_df)

                                    result_id = f"NUM_{var}"
                                    st.session_state["saved_tables"][result_id] = num_df
                                    st.session_state["result_cart"].append(
                                        CandidateResult(
                                            id=result_id, title=f"Đặc điểm định lượng của biến {var}",
                                            result_type="baseline", variables=[var],
                                            scientific_value=3.5, clinical_importance=4.0, discussion_value=3.0
                                        )
                                    )
                            st.success(f"✅ Đã nạp {len(num_vars)} bảng định lượng vào Giỏ!")

                st.write("---")

                # ==========================================
                # 3. BẢNG CHÉO & KIỂM ĐỊNH
                # ==========================================
                st.subheader("3. Bảng chéo và kiểm định (Chi-square / Fisher / OR)")
                cc1, cc2 = st.columns(2)
                with cc1:
                    select_all_deps = st.checkbox("✅ Chọn tất cả biến phụ thuộc", key=ui_key("chk_all_deps"))
                    deps_widget_key = ui_key("cross_deps_all") if select_all_deps else ui_key("cross_deps_manual")
                    deps = st.multiselect(
                        "Các biến phụ thuộc", all_cols,
                        default=all_cols if select_all_deps else [],
                        key=deps_widget_key,
                    )
                with cc2:
                    select_all_indeps = st.checkbox("✅ Chọn tất cả biến độc lập", key=ui_key("chk_all_indeps"))
                    indeps_widget_key = ui_key("cross_indeps_all") if select_all_indeps else ui_key("cross_indeps_manual")
                    indeps = st.multiselect(
                        "Các biến độc lập cần đối chiếu", all_cols,
                        default=all_cols if select_all_indeps else [],
                        key=indeps_widget_key,
                    )

                if st.button("Quét Crosstab + Kiểm định (Nạp TẤT CẢ vào Giỏ)", key="calc_cross"):
                    if not deps or not indeps:
                        st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
                    else:
                        found_count = 0
                        processed_pairs = set()
                        with st.spinner("Đang cày xới toàn bộ ma trận số liệu..."):
                            for dep in deps:
                                for indep in indeps:
                                    if dep == indep:
                                        continue
                                    pair_key = frozenset([dep, indep])
                                    if pair_key in processed_pairs:
                                        continue
                                    processed_pairs.add(pair_key)

                                    try:
                                        result = crosstab_test(df, indep, dep)
                                        found_count += 1

                                        sig_marker = "🟢 CÓ Ý NGHĨA" if result['p_value'] < 0.05 else "⚪ KHÔNG Ý NGHĨA"
                                        st.markdown(f"**► Mối liên quan giữa: [{indep}] & [{dep}] — {sig_marker}**")
                                        st.dataframe(result["table"])
                                        st.write(f"- **Kiểm định:** {result['test']} | **p-value:** `{result['p_value']:.4g}`")

                                        if "effect_size" in result:
                                            st.write(f"- **Chỉ số (Effect Size):** `{result['effect_size']}`")

                                        result_id = f"CROSS_{indep}_{dep}"
                                        st.session_state["saved_tables"][result_id] = result["table"]
                                        st.session_state["result_cart"].append(
                                            CandidateResult(
                                                id=result_id, title=f"Mối liên quan giữa {indep} và {dep}",
                                                result_type="association", variables=[indep, dep],
                                                p_value=result['p_value'],
                                                scientific_value=4.5, clinical_importance=4.5, discussion_value=5.0
                                            )
                                        )
                                    except Exception as e:
                                        st.error(f"⚠️ Lỗi phân tích chéo [{indep} & {dep}]: {str(e)}")

                        if found_count > 0:
                            st.success(f"✅ Đã phân tích và nạp {found_count} bảng kiểm định vào Giỏ!")

                st.write("---")

                # ==========================================
                # 4. SO SÁNH 2 NHÓM
                # ==========================================
                st.subheader("4. So sánh biến định lượng giữa 2 nhóm (T-test / Mann-Whitney)")
                gc1, gc2 = st.columns(2)
                with gc1:
                    select_all_groups = st.checkbox("✅ Chọn tất cả biến nhóm", key=ui_key("chk_all_groups"))
                    group_vars_widget_key = ui_key("group_vars_all") if select_all_groups else ui_key("group_vars_manual")
                    group_vars = st.multiselect(
                        "Biến nhóm (Tự lọc biến 2 mức)", all_cols,
                        default=all_cols if select_all_groups else [],
                        key=group_vars_widget_key,
                    )
                with gc2:
                    valid_num_cols = numeric_candidates if 'numeric_candidates' in locals() and numeric_candidates else all_cols
                    select_all_vals = st.checkbox("✅ Chọn tất cả biến định lượng", key=ui_key("chk_all_vals"))
                    val_vars_widget_key = ui_key("val_vars_all") if select_all_vals else ui_key("val_vars_manual")
                    val_vars = st.multiselect(
                        "Biến định lượng cần so sánh", valid_num_cols,
                        default=valid_num_cols if select_all_vals else [],
                        key=val_vars_widget_key,
                    )

                if st.button("Quét kiểm định so sánh (Nạp TẤT CẢ vào Giỏ)", key="run_group_compare"):
                    if not group_vars or not val_vars:
                        st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
                    else:
                        found_count = 0
                        with st.spinner("Đang rà soát và tính toán..."):
                            for gv in group_vars:
                                for vv in val_vars:
                                    if gv == vv:
                                        continue
                                    try:
                                        result = compare_two_groups(df, gv, vv)
                                        found_count += 1

                                        sig_marker = "🟢 KHÁC BIỆT" if result['p_value'] < 0.05 else "⚪ TƯƠNG ĐỒNG"
                                        g1n, g2n = result["group_names"]
                                        st.markdown(f"**► Sự phân bố của [{vv}] giữa 2 nhóm [{gv}] — {sig_marker}**")

                                        st.write(f"- **Kiểm định:** {result['test']} | **p-value:** `{result['p_value']:.4g}`")
                                        if "effect_size" in result:
                                            st.write(f"- **Chỉ số (Effect Size):** `{result['effect_size']}`")

                                        comp_df = pd.DataFrame({
                                            g1n: [result["group1_stats"]],
                                            g2n: [result["group2_stats"]],
                                        }, index=["Giá trị"])
                                        st.dataframe(comp_df)

                                        result_id = f"COMP_{gv}_{vv}"
                                        st.session_state["saved_tables"][result_id] = comp_df
                                        st.session_state["result_cart"].append(
                                            CandidateResult(
                                                id=result_id, title=f"Sự khác biệt của biến {vv} giữa các nhóm {gv}",
                                                result_type="association", variables=[gv, vv],
                                                p_value=result['p_value'],
                                                scientific_value=4.5, clinical_importance=4.5, discussion_value=5.0
                                            )
                                        )
                                    except Exception as e:
                                        st.error(f"⚠️ Lỗi so sánh [{gv} & {vv}]: {str(e)}")

                        if found_count > 0:
                            st.success(f"✅ Đã nạp {found_count} kết quả so sánh vào Giỏ!")

                st.write("---")

                # ==========================================
                # 5. HỒI QUY LOGISTIC
                # ==========================================
                st.subheader("5. Hồi quy logistic nhị phân (OR và 95% CI)")
                outcome_candidates = [c for c in all_cols if df[c].dropna().nunique() == 2]
                forbidden_keywords = ["unnamed", "ngay", "ngày", "ten", "tên", "ma", "mã", "sobenhan", "id"]
                predictor_candidates = [
                    c for c in all_cols
                    if not any(kw in str(c).lower() for kw in forbidden_keywords)
                    and df[c].dropna().nunique() > 1
                ]

                lc1, lc2 = st.columns([1, 2])
                with lc1:
                    select_all_outcomes = st.checkbox("✅ Chọn tất cả biến kết cục", key=ui_key("chk_all_outcomes"))
                    outcomes_widget_key = ui_key("log_outcomes_all") if select_all_outcomes else ui_key("log_outcomes_manual")
                    outcomes = st.multiselect(
                        "Biến kết cục (Nhị phân)", outcome_candidates,
                        default=outcome_candidates if select_all_outcomes else [],
                        key=outcomes_widget_key,
                    )
                with lc2:
                    select_all_predictors = st.checkbox("✅ Chọn tất cả yếu tố dự báo", key=ui_key("chk_all_predictors"))
                    predictors_widget_key = ui_key("log_predictors_all") if select_all_predictors else ui_key("log_predictors_manual")
                    predictors = st.multiselect(
                        "Yếu tố dự báo", predictor_candidates,
                        default=predictor_candidates if select_all_predictors else [],
                        key=predictors_widget_key,
                    )

                if st.button("Chạy Logistic Regression đa biến (Nạp vào Giỏ)", key="run_logistic"):
                    if not outcomes or not predictors:
                        st.warning("Chọn ít nhất một biến ở mỗi bên.")
                    else:
                        found_count = 0
                        with st.spinner("Đang xây dựng mô hình hồi quy..."):
                            for out in outcomes:
                                preds = [p for p in predictors if p != out]
                                if not preds:
                                    continue
                                try:
                                    result_df, summary = binary_logistic_regression(df, out, preds)
                                    found_count += 1

                                    st.markdown(f"**► MÔ HÌNH HỒI QUY ĐA BIẾN CHO KẾT CỤC: [{out}]**")
                                    st.info(summary)
                                    st.dataframe(result_df)

                                    result_id = f"LOG_{out}"
                                    st.session_state["saved_tables"][result_id] = result_df
                                    st.session_state["result_cart"].append(
                                        CandidateResult(
                                            id=result_id, title=f"Mô hình hồi quy logistic đánh giá yếu tố liên quan đến {out}",
                                            result_type="regression", variables=[out] + preds,
                                            scientific_value=5.0, clinical_importance=5.0, discussion_value=5.0
                                        )
                                    )
                                except Exception as e:
                                    st.error(f"⚠️ Không thể xây dựng mô hình cho [{out}]: {str(e)}")

                        if found_count > 0:
                            st.success(f"✅ Đã nạp {found_count} mô hình hồi quy vào Giỏ!")

                st.write("---")

                # ==========================================
                # 8. DIỄN GIẢI BẰNG AI
                # ==========================================
                st.subheader("8. Diễn giải kết quả bằng AI (không tính lại số liệu)")
                st.info("💡 **Không cần copy/paste thủ công!** Anh chỉ cần chọn bảng đã tính ở trên từ danh sách, hệ thống sẽ tự động rút số liệu chuyển cho AI.")

                saved_tables = st.session_state.get("saved_tables", {})
                table_options = ["-- Chỉ dùng số liệu dán tay bên dưới --"] + list(saved_tables.keys())

                selected_table_key = st.selectbox(
                    "Lựa chọn bảng từ Giỏ kết quả:",
                    options=table_options,
                    key=ui_key("select_ai_table")
                )

                interpretation_request = st.text_area(
                    "Hoặc dán số liệu bổ sung vào đây (nếu có):",
                    height=120,
                    key=ui_key("interpretation_request")
                )

                if st.button("🤖 AI diễn giải", type="primary", key="ai_interpret"):
                    final_data = ""

                    if selected_table_key != "-- Chỉ dùng số liệu dán tay bên dưới --":
                        df_target = saved_tables[selected_table_key]
                        final_data += f"TÊN BẢNG: {selected_table_key}\n"
                        final_data += df_target.to_markdown(index=False) + "\n\n"

                    if interpretation_request.strip():
                        final_data += f"SỐ LIỆU BỔ SUNG:\n{interpretation_request.strip()}"

                    if not final_data.strip():
                        st.warning("⚠️ Anh chưa chọn bảng nào hoặc chưa dán số liệu!")
                    else:
                        try:
                            prompt = f"{BASE_SYSTEM_RULES}\nBạn chỉ được DIỄN GIẢI kết quả thống kê dưới đây. Không được tính lại hoặc sửa số liệu.\nKẾT QUẢ:\n{final_data}"

                            with st.spinner("AI đang đọc và diễn giải số liệu..."):
                                output = call_gemini(prompt, model=st.session_state.get("selected_model", DEFAULT_MODEL))
                                if output:
                                    st.markdown(output)
                        except Exception as exc:
                            st.error(f"Lỗi gọi AI: {exc}")

            except Exception as exc:
                st.error(f"Lỗi khi đọc hoặc xử lý file Excel: {exc}")

    # ------------------------------------------------------------
    # TAB 5 – Audit
    # ------------------------------------------------------------
    with tabs[4]:
        st.header("🔎 Audit luận văn toàn diện")
        st.markdown(
            '<div class="warning-box">⚠️ <b>Giới hạn cần biết:</b> Các công cụ ở '
            "tab này chỉ đưa ra <b>chỉ báo nguy cơ / gợi ý Audit thêm</b>. "
            "Không có công cụ nào khẳng định chắc chắn 100%.</div>",
            unsafe_allow_html=True,
        )

        Audit_text = st.text_area("Dán đoạn văn cần Audit vào đây:", height=250, key=ui_key("Audit_text"))

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        st.write("---")
        ket_qua_Audit_container = st.container()

        # 1. Audit SỐ LIỆU
        with c1:
            if st.button("🔢 Số liệu", key="Audit_numbers", use_container_width=True):
                if not Audit_text.strip():
                    st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang truy xuất và đối chiếu..."):
                        result = Audit_generated_text(Audit_text)
                    with ket_qua_Audit_container:
                        st.markdown("### 🔢 Kết quả Audit Số liệu (3 Cấp Độ)")
                        st.success(f"**Level 1 (Khớp chính xác):** {', '.join(result['exact_matches']) if result['exact_matches'] else 'Không có'}")
                        st.info(f"**Level 2 (Khớp phái sinh):** {', '.join(result['derived_matches']) if result['derived_matches'] else 'Không có'}")
                        if result["warnings"]:
                            st.error(f"**Level 3 (⚠️ SỐ LIỆU LẠ):** {', '.join(result['warnings'])}")
                            st.caption("Cần đối chiếu lại bản gốc!")
                        else:
                            st.success("**Level 3:** Không phát hiện số liệu lạ!")
                        with st.expander("📄 Xem bằng chứng đối chiếu"):
                            for ev in result["evidence_used"]:
                                st.caption(f"Nguồn: {ev['file_name']} (Trang {ev['page']})")
                                st.write(f"> {ev['text']}")

        # 2. Audit TRÍCH DẪN
        with c2:
            if st.button("📚 Trích dẫn", key="Audit_citation", use_container_width=True):
                if not Audit_text.strip():
                    st.warning("Chưa có văn bản.")
                else:
                    citations_in_text = re.findall(r"\[(\d+)\]", Audit_text)
                    current_refs = {str(ref['vancouver_index']): ref for ref in st.session_state.get("current_references", [])}
                    with ket_qua_Audit_container:
                        st.markdown("### 📚 Kết quả Audit Citation Validator")
                        if not citations_in_text:
                            st.info("Không tìm thấy trích dẫn định dạng [n] trong văn bản.")
                        else:
                            fake_citations = [c for c in citations_in_text if c not in current_refs]
                            if fake_citations:
                                st.error(f"❌ Phát hiện trích dẫn ẢO: [{'], ['.join(fake_citations)}]")
                            else:
                                st.success("✅ Toàn bộ trích dẫn khớp với danh mục hiện tại!")
                            with st.expander("Nguồn của các trích dẫn trong bài"):
                                for c in citations_in_text:
                                    if c in current_refs:
                                        meta = current_refs[c]['metadata']
                                        st.write(f"**[{c}]** {meta.get('authors', 'N/A')}. {meta.get('title', 'N/A')} ({meta.get('year', 'N/A')})")

        # 3. TRÙNG LẶP NỘI BỘ
        with c3:
            if st.button("🔍 Trùng lặp", key="Audit_overlap", use_container_width=True):
                if not Audit_text.strip():
                    st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang quét Jaccard Similarity..."):
                        overlaps = internal_overlap_Audit(Audit_text)
                    with ket_qua_Audit_container:
                        st.markdown("### 🔍 Kết quả tìm trùng lặp nội bộ")
                        if not overlaps:
                            st.info("Không tìm thấy đoạn trùng đáng kể trong kho tài liệu hiện tại.")
                        else:
                            for item in overlaps:
                                st.markdown(f"**{item['file']} – trang {item['page']}**\nSimilarity: **{item['similarity']}**\n\n> {item['text']}")

        # 4. Audit CHÍNH TẢ
        with c4:
            if st.button("🔤 Chính tả", key="Audit_spelling", use_container_width=True):
                if not Audit_text.strip():
                    st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang rà soát lỗi chính tả & thuật ngữ..."):
                        response = spelling_and_terminology_check(Audit_text)
                    with ket_qua_Audit_container:
                        st.markdown("### 🔤 Audit Chính tả & Thuật ngữ")
                        st.markdown(response if response else "Không nhận được phản hồi từ AI.")

        # 5. Audit VĂN PHONG AI
        with c5:
            if st.button("🤖 Check văn AI", key="Audit_ai_style", use_container_width=True):
                if not Audit_text.strip():
                    st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang phân tích dấu hiệu văn phong máy móc..."):
                        style_analysis = heuristic_ai_style_score(Audit_text)
                    with ket_qua_Audit_container:
                        st.markdown("### 🤖 Chỉ báo nguy cơ văn bản do AI viết")
                        st.markdown(style_analysis if style_analysis else "Không nhận được phản hồi từ AI.")

        # 6. PHẢN BIỆN LOGIC
        with c6:
            if st.button("⚖️ Phản biện", key="logic_review", use_container_width=True):
                if not Audit_text.strip():
                    st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang soi logic nghiên cứu..."):
                        prompt = f"""
                        {BASE_SYSTEM_RULES}
                        Đóng vai phản biện luận văn CKI Dược lâm sàng. Đọc đoạn văn sau và chỉ ra các điểm yếu logic:
                        1. Có khẳng định nào thiếu bằng chứng không?
                        2. Có nhảy từ tương quan sang nhân quả không?
                        3. Kết luận có vượt quá giới hạn thiết kế nghiên cứu không?
                        ĐOẠN VĂN GỐC: {Audit_text}
                        """
                        response = call_gemini(prompt)
                    with ket_qua_Audit_container:
                        st.markdown("### ⚖️ Kết quả Phản biện Logic")
                        st.markdown(response if response else "Không nhận được phản hồi từ AI.")

    # ------------------------------------------------------------
    # TAB 6 – NGUỒN, CẤU HÌNH & QUẢN LÝ CHECKPOINT
    # ------------------------------------------------------------
    with tabs[5]:
        st.header("⚙️ Nguồn, citation và cấu hình")
        st.write(f"**Gemini model:** `{DEFAULT_MODEL}`")
        st.write(f"**Embedding model:** `{DEFAULT_EMBEDDING}`")
        st.write(f"**SerpAPI (tra cứu tạp chí VN):** {'✅ đã cấu hình' if get_serpapi_key() else '❌ chưa cấu hình (tùy chọn)'}")

        st.write("---")
        
        # ============================================================
        # TÍNH NĂNG MỚI: AI QUÉT VÀ LƯU METADATA HÀNG LOẠT
        # ============================================================
        st.subheader("🤖 AI Tự động quét và lưu Metadata hàng loạt")
        st.info("💡 Bấm nút dưới đây để AI tự động quét trang đầu của toàn bộ file PDF trong kho và điền thông tin (Tác giả, Năm, Tên bài...).")
        if st.button("🚀 AI Tự động đọc toàn bộ file PDF và lưu Metadata", type="primary"):
            docs = st.session_state.get("documents", {})
            chunks = st.session_state.get("chunks", [])
            count_updated = 0
            
            if not docs:
                st.warning("⚠️ Evidence Database đang trống, chưa có tài liệu nào!")
            else:
                with st.spinner("AI đang quét trang đầu của tất cả các file PDF..."):
                    for source_id, meta in docs.items():
                        if meta.get("origin") == "PDF":
                            # Tìm đoạn văn bản (chunk) đầu tiên của file này để đưa cho AI đọc
                            chunk_text = ""
                            for c in chunks:
                                c_sid = c.get("source_id") if isinstance(c, dict) else getattr(c, 'source_id', None)
                                if c_sid == source_id:
                                    chunk_text = c.get("text") if isinstance(c, dict) else c.text
                                    break
                            
                            if chunk_text:
                                meta_ai = extract_metadata_from_text_ai(chunk_text)
                                if meta_ai:
                                    # Cập nhật thông tin mới
                                    meta["title"] = meta_ai.get("title") or meta.get("title", "")
                                    meta["authors"] = meta_ai.get("authors") or meta.get("authors", "")
                                    meta["year"] = meta_ai.get("year") or meta.get("year", "")
                                    meta["journal"] = meta_ai.get("journal") or meta.get("journal", "")
                                    meta["doi"] = meta_ai.get("doi") or meta.get("doi", "")
                                    meta["pmid"] = meta_ai.get("pmid") or meta.get("pmid", "")
                                    
                                    st.session_state["documents"][source_id] = meta
                                    count_updated += 1
                                    
                if count_updated > 0:
                    st.success(f"✅ Đã tự động cập nhật metadata cho {count_updated} file PDF!")
                    st.rerun()
                else:
                    st.warning("Không tìm thấy dữ liệu văn bản để quét.")

        st.write("---")
        st.subheader("Danh sách domain tạp chí Y học Việt Nam (dùng cho Tab 2)")
        domains_text = st.text_area(
            "Mỗi domain một dòng:",
            value="\n".join(st.session_state["vn_journal_domains"]),
            height=120,
            key=ui_key("domains_text_area")
        )
        if st.button("💾 Lưu danh sách domain"):
            st.session_state["vn_journal_domains"] = [
                d.strip() for d in domains_text.splitlines() if d.strip()
            ]
            st.success("Đã cập nhật.")

        st.write("---")
        st.subheader("Citation Registry & Metadata thủ công")
        
        registry = st.session_state["citation_registry"]
        if registry:
            registry_rows = []
            for source_id, number in sorted(registry.items(), key=lambda x: x[1]):
                meta = source_metadata(source_id)
                registry_rows.append({
                    "Citation": f"[{number}]", "Source ID": source_id,
                    "Nguồn gốc": meta.get("origin", ""), "File/Tiêu đề": meta.get("file_name", ""),
                    "Tác giả": meta.get("authors", ""), "Năm": meta.get("year", ""),
                    "Tạp chí": meta.get("journal", ""), "DOI": meta.get("doi", ""),
                    "PMID": meta.get("pmid", ""),
                })
            st.dataframe(pd.DataFrame(registry_rows))

            st.subheader("Danh mục tham khảo hiện tại")
            st.code(citation_bibliography(), language="text")
        else:
            st.info("Citation registry chưa có dữ liệu.")

        st.write("---")
        
        # Danh sách chi tiết từng nguồn
        for source_id, meta in list(st.session_state["documents"].items()):
            with st.expander(f"[{meta.get('origin','')}] {source_id} – {meta.get('file_name', '')}"):
                
                # Nút AI đọc riêng lẻ cho file này (vẫn giữ để anh sửa từng file nếu cần)
                if meta.get("origin") == "PDF":
                    if st.button(f"🤖 Nhờ AI đọc lại file này", key=f"ai_meta_{source_id}"):
                        chunk_text = ""
                        for c in st.session_state["chunks"]:
                            if (isinstance(c, dict) and c.get("source_id") == source_id) or (hasattr(c, 'source_id') and c.source_id == source_id):
                                chunk_text = c.get("text") if isinstance(c, dict) else c.text
                                break
                        if chunk_text:
                            meta_ai = extract_metadata_from_text_ai(chunk_text)
                            if meta_ai:
                                meta.update(meta_ai)
                                st.session_state["documents"][source_id] = meta
                                st.success("✅ Đã cập nhật!")
                                st.rerun()
                            else:
                                st.error("AI không tìm thấy thông tin.")
                
                # --- Khu vực nhập liệu tay ---
                mc1, mc2 = st.columns(2)
                with mc1:
                    authors = st.text_input("Tác giả", value=meta.get("authors", ""), key=ui_key(f"authors_{source_id}"))
                    title = st.text_input("Tên bài/tài liệu", value=meta.get("title", ""), key=ui_key(f"title_{source_id}"))
                    year = st.text_input("Năm", value=meta.get("year", ""), key=ui_key(f"year_{source_id}"))
                with mc2:
                    journal = st.text_input("Tạp chí/NXB", value=meta.get("journal", ""), key=ui_key(f"journal_{source_id}"))
                    doi = st.text_input("DOI", value=meta.get("doi", ""), key=ui_key(f"doi_{source_id}"))
                    pmid = st.text_input("PMID", value=meta.get("pmid", ""), key=ui_key(f"pmid_{source_id}"))

                if st.button("💾 Lưu metadata", key=f"save_{source_id}"):
                    meta.update({
                        "authors": authors, "title": title, "year": year,
                        "journal": journal, "doi": doi, "pmid": pmid,
                    })
                    st.session_state["documents"][source_id] = meta
                    st.success("Đã lưu.")

        # ============================================================
        # PHẦN CUỐI: NGUYÊN TẮC & EXPORT
        # ============================================================
        st.write("---")
        st.subheader("Nguyên tắc sử dụng")
        st.markdown("""
            * Không xem bản nháp AI là kết quả cuối cùng.
            * Không dùng citation nếu chưa truy ngược được về nguồn.
            * Người nghiên cứu phải Audit bản gốc trước khi chấp nhận số liệu và diễn giải.
            """)

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            if st.button("📄 Xuất bản nháp hiện tại ra Word", key="export_word", use_container_width=True):
                if not st.session_state["last_generated"]:
                    st.warning("Chưa có bản nháp.")
                else:
                    docx_data = create_word_document(
                        title="Bản nháp hỗ trợ nghiên cứu",
                        body=st.session_state["last_generated"],
                        bibliography=citation_bibliography(),
                    )
                    st.download_button("📥 Tải Word", data=docx_data, file_name="Ban_nhap.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        with c_exp2:
            if st.button("🧹 Làm mới Giao diện", key=ui_key("btn_reset_ui"), use_container_width=True):
                reset_ui_state()
                st.rerun()
       
        # ============================================================
        # QUẢN LÝ CHECKPOINT & DỰ ÁN LUẬN VĂN (XUẤT/NHẬP FILE JSON)
        # ============================================================
        st.write("---")
        st.subheader("💾 Lưu trữ dự án dài hạn")
        st.info(
            "Vì máy chủ đám mây sẽ xóa dữ liệu khi khởi động lại, anh hãy **Tải file dự án (.json)** về máy sau mỗi phiên làm việc. "
            "Lần sau vào lại, chỉ cần tải file đó lên để khôi phục toàn bộ 100% trạng thái (PDF, Metadata, Kết quả thống kê)."
        )

        col_ex1, col_ex2 = st.columns(2)
        
        # --- 1. XUẤT DỰ ÁN (EXPORT) ---
        with col_ex1:
            st.markdown("#### 📥 1. Sao lưu dự án ra máy")
            # Đóng gói dữ liệu
            project_data = {
                "documents": st.session_state.get("documents", {}),
                "chunks": st.session_state.get("chunks", []),
                "citation_registry": st.session_state.get("citation_registry", {}),
                "current_references": st.session_state.get("current_references", []),
                "result_cart": [vars(item) if hasattr(item, "__dict__") else item for item in st.session_state.get("result_cart", [])],
                # Convert DataFrame sang dict để lưu JSON
                "saved_tables": {k: v.to_dict(orient='split') for k, v in st.session_state.get("saved_tables", {}).items()}
            }
            json_str = json.dumps(project_data, ensure_ascii=False, indent=4)
            
            st.download_button(
                label="📥 Tải file dự án (.json)",
                data=json_str,
                file_name="Du_An_Luan_Van.json",
                mime="application/json",
                use_container_width=True
            )

        # --- 2. NHẬP DỰ ÁN (IMPORT) ---
        with col_ex2:
            st.markdown("#### 📂 2. Khôi phục dự án từ máy")
            uploaded_proj_file = st.file_uploader("Chọn file .json đã lưu:", type=["json"], key=ui_key("upload_proj_json"))
            
            if uploaded_proj_file is not None:
                if st.button("🚀 Khôi phục dự án", type="primary", use_container_width=True):
                    try:
                        loaded_data = json.load(uploaded_proj_file)
                        
                        # Phục hồi dữ liệu
                        st.session_state["documents"] = loaded_data.get("documents", {})
                        st.session_state["chunks"] = loaded_data.get("chunks", [])
                        st.session_state["citation_registry"] = loaded_data.get("citation_registry", {})
                        st.session_state["current_references"] = loaded_data.get("current_references", [])
                        
                        # Phục hồi Giỏ kết quả
                        cart_raw = loaded_data.get("result_cart", [])
                        st.session_state["result_cart"] = [CandidateResult(**item) if isinstance(item, dict) else item for item in cart_raw]
                        
                        # Phục hồi Saved Tables
                        tables_raw = loaded_data.get("saved_tables", {})
                        st.session_state["saved_tables"] = {k: pd.DataFrame.from_dict(v, orient='split') for k, v in tables_raw.items()}
                        
                        # Tái thiết lập bộ chỉ mục tìm kiếm (RAG)
                        with st.spinner("Đang xây dựng lại index tìm kiếm..."):
                            rebuild_index()
                        
                        st.success("🎉 Khôi phục thành công! Mọi tài liệu và kết quả đã sẵn sàng.")
                        time.sleep(1)
                        reset_ui_state()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Lỗi khi đọc file: {str(e)}")

        st.caption("⚠️ **Lưu ý:** Luôn tải bản sao lưu (.json) về máy tính cá nhân sau mỗi lần làm việc quan trọng.")

if __name__ == "__main__":
    main()
