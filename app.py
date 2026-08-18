# app.py
# ============================================================
# HỖ TRỢ NGHIÊN CỨU KHOA HỌC – EVIDENCE-BASED RAG
# Bản tối ưu cho luận văn Chuyên khoa cấp I – Dược lâm sàng
# ============================================================

import io
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Google Gemini SDK mới: pip install google-genai
from google import genai
from google.genai import types

# Embedding: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

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

# 4. Import bộ máy Quản lý Checkpoint / Dự án (lưu & khôi phục xuống đĩa)
from project_storage import save_project, load_project, list_projects, delete_project

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
# 2. CSS – GIAO DIỆN HỌC THUẬT, TỐI GIẢN & TỐI ƯU HIỆU NĂNG
# ============================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ===== NỀN TOÀN TRANG: XÁM SÁNG TRUNG TÍNH ===== */
    .stApp {
        background-color: #f8f9fa;
        color: #2c3e50;
    }

    /* ===== TIÊU ĐỀ CHÍNH ===== */
    h1 {
        color: #1e293b !important;
        text-align: center;
        font-weight: 800;
        margin-top: 0 !important;
        margin-bottom: 6px;
    }
    h2, h3, h4 { color: #0f172a !important; font-weight: 600; }

    /* ===== KHỐI NỘI DUNG TAB - TRẮNG PHẲNG, VIỀN MỎNG ===== */
    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 32px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }

    /* ===== THANH TAB ===== */
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

    /* ===== NÚT BẤM - PHẲNG, MÀU TINH TẾ ===== */
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
    
    /* Nút Primary (Màu xanh Blue học thuật) */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    /* ===== BẢNG DỮ LIỆU ===== */
    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    /* ===== Ô NHẬP LIỆU ===== */
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

    /* ===== EXPANDER / CONTAINER / ALERT ===== */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 6px;
        font-weight: 600;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    .stAlert { border-radius: 8px !important; }

    /* ===== CÁC KHỐI CẢNH BÁO TÙY CHỈNH (MÀU PASTEL NHẸ) ===== */
    .warning-box { border-left: 4px solid #f59e0b; padding: 12px 16px; background: #fffbeb; border-radius: 6px; color: #92400e; font-size: 0.95rem;}
    .danger-box  { border-left: 4px solid #ef4444; padding: 12px 16px; background: #fef2f2; border-radius: 6px; color: #991b1b; font-size: 0.95rem;}
    .success-box { border-left: 4px solid #10b981; padding: 12px 16px; background: #ecfdf5; border-radius: 6px; color: #065f46; font-size: 0.95rem;}

    /* ===== VĂN BẢN HỌC THUẬT DO AI TẠO - CANH ĐỀU 2 LỀ, DỄ ĐỌC ===== */
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
# 4. SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "documents": {},            # source_id -> SourceDocument dict
        "chunks": [],               # list[EvidenceChunk dict]
        "embeddings": None,         # np.ndarray
        "citation_registry": {},    # source_id -> citation number
        "audit_log": [],
        "last_generated": "",
        "last_evidence": [],
        "vn_journal_domains": list(DEFAULT_VN_JOURNAL_DOMAINS),
        # Tab 3 - tra cứu đa nguồn
        "t3_pm_data": [],
        "t3_vn_data": [],
        "t3_en_keyword": "",
        "t3_query": "",
        # Tab 4 - Result Database / cấu trúc Chương 3
        "result_cart": [],
        "saved_tables": {},
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

            # Bắt chung cả lỗi 429 (Hết Quota) và 503 (Server quá tải)
            if any(code in error_msg for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                if attempt < max_retries - 1:
                    wait_time = 15  # Cho ứng dụng nghỉ 15 giây để server Google hạ nhiệt
                    status = st.warning(f"⏳ Trạm máy chủ Google đang quá tải đột xuất. Ứng dụng tự động đợi {wait_time} giây rồi thử lại (Lần {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    status.empty()  # Xóa câu thông báo sau khi chờ xong
                else:
                    st.error("❌ Máy chủ Google Gemini hiện đang quá bận. Anh vui lòng đợi 1-2 phút rồi bấm thử lại nhé!")
                    return None
            else:
                # Nếu là các lỗi mạng khác, đợi 3 giây rồi thử lại
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
    new_sources, new_chunks_count, errors = 0, 0, []
    new_chunks_list = [] # Lưu trữ các chunk mới để update index

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
        # Truyền danh sách chunk mới vào để hệ thống chỉ encode phần mới
        rebuild_index(new_chunks=new_chunks_list)

    return new_sources, new_chunks_count, errors

# ============================================================
# 8. TRA CỨU TỪ KHÓA (ĐÃ TỐI ƯU HÓA CHO PUBMED)
# ============================================================

def translate_query_to_mesh(vietnamese_query: str) -> str:
    prompt = f"""
    Bạn là một chuyên gia tra cứu tài liệu y khoa. 
    Nhiệm vụ: Chuyển đổi tên đề tài tiếng Việt sau thành chuỗi từ khóa tiếng Anh hiệu quả nhất để tra cứu trên PubMed.
    
    NGUYÊN TẮC BẮT BUỘC:
    1. Trích xuất các danh từ chính (Ví dụ: Vancomycin, drug utilization, pharmacokinetics).
    2. Dùng toán tử AND, OR để kết nối.
    3. TUYỆT ĐỐI KHÔNG dùng cấu trúc MeSH chứa dấu gạch chéo (ví dụ KHÔNG dùng: Vancomycin/therapeutic use). Việc ép subheadings sẽ làm mất kết quả tìm kiếm.
    4. Giữ chuỗi tìm kiếm ngắn gọn, từ 2 đến 4 cụm từ khóa.
    
    Từ gốc tiếng Việt: "{vietnamese_query}"
    
    Chỉ trả về chuỗi tiếng Anh, không giải thích, không dùng markdown.
    """
    
    # Sử dụng MODEL_LITE để dịch thuật nhanh
    text = call_gemini(prompt, model=MODEL_LITE)
    if text:
        # Dọn dẹp các ký tự thừa nếu AI lỡ tay sinh ra
        return text.strip().strip('"').strip("'").replace("\n", " ")
    
    return vietnamese_query
def extract_vn_keywords(vietnamese_query: str) -> str:
    """Rút gọn câu dài thành từ khóa tiếng Việt cốt lõi để tìm Google Scholar dễ ra bài hơn"""
    prompt = f"""
    Bạn là chuyên gia tra cứu tài liệu. 
    Nhiệm vụ: Rút gọn tên đề tài sau thành 1-2 từ khóa quan trọng nhất bằng tiếng Việt.
    KHÔNG dùng câu dài. CHỈ giữ lại danh từ cốt lõi (hoạt chất, tên bệnh).
    
    Đề tài: "{vietnamese_query}"
    
    Chỉ trả về từ khóa rút gọn, không giải thích.
    """
    text = call_gemini(prompt, model=MODEL_LITE)
    if text:
        return text.strip().strip('"').strip("'").replace("\n", " ")
    return vietnamese_query

# ============================================================
# 10. INDEX / VECTOR RETRIEVAL (HYBRID: EMBEDDING + BM25)
# ============================================================
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Any, Dict, List
import streamlit as st

def evidence_database_summary() -> Dict[str, Any]:
    """Đếm số nguồn/số đoạn bằng chứng theo từng nguồn gốc (PDF/PubMed/VN/Thủ công)."""
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
    """Hiển thị hộp trạng thái Evidence Database - dùng ở đầu các tab cần bằng chứng."""
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


def rebuild_index(new_chunks: List[Dict[str, Any]] = None):
    """
    Cải tiến P1: Incremental Indexing & BM25
    Chỉ mã hóa (embed) những đoạn văn MỚI được thêm vào thay vì làm lại từ đầu.
    Đồng thời xây dựng lại bộ từ điển BM25 cho tìm kiếm từ khóa.
    """
    all_chunks = st.session_state.get("chunks", [])
    if not all_chunks:
        st.session_state["embeddings"] = None
        st.session_state["bm25"] = None
        return

    # 1. CẬP NHẬT EMBEDDING (Vector Ngữ nghĩa) - Chỉ mã hóa phần mới nếu đã có index cũ
    if new_chunks and st.session_state.get("embeddings") is not None:
        new_texts = [c["text"] for c in new_chunks]
        new_matrix = get_embeddings(new_texts)
        st.session_state["embeddings"] = np.vstack([st.session_state["embeddings"], new_matrix])
    else:
        texts = [c["text"] for c in all_chunks]
        st.session_state["embeddings"] = get_embeddings(texts)

    # 2. XÂY DỰNG TỪ ĐIỂN BM25 (Từ khóa chính xác)
    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    st.session_state["bm25"] = BM25Okapi(tokenized_corpus)


def retrieve_evidence(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    chunks = st.session_state.get("chunks", [])
    matrix = st.session_state.get("embeddings")
    bm25 = st.session_state.get("bm25")

    if not chunks or matrix is None or bm25 is None:
        return []

    # 1. Lấy Top 30 ứng viên từ Semantic Search
    query_vector = get_embeddings([query])[0]
    semantic_scores = matrix @ query_vector
    sem_indices = np.argsort(semantic_scores)[::-1][:30]
    
    # 2. Lấy Top 30 ứng viên từ BM25 Keyword Search
    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    bm25_indices = np.argsort(bm25_scores)[::-1][:30]

    # 3. Tổng hợp bằng Reciprocal Rank Fusion (RRF constant c = 60)
    rrf_scores = {}
    c = 60.0

    for rank, idx in enumerate(sem_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (0.65 / (c + rank + 1))

    for rank, idx in enumerate(bm25_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (0.35 / (c + rank + 1))

    # Sắp xếp lại theo điểm RRF tích hợp cao nhất
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    k = min(k, len(sorted_indices))
    final_indices = sorted_indices[:k]

    results = []
    for idx in final_indices:
        item = dict(chunks[idx])
        item["score"] = float(rrf_scores[idx])
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
# 11. CITATION ENGINE (CẤU TRÚC MỚI CHUẨN VANCOUVER)
# ============================================================
from citation_engine import CitationEngine

def get_citation_engine() -> CitationEngine:
    """Khởi tạo và lấy bộ máy Citation Engine từ Session State"""
    if "citation_engine" not in st.session_state:
        st.session_state["citation_engine"] = CitationEngine()
    return st.session_state["citation_engine"]

def source_metadata(source_id: str) -> Dict[str, Any]:
    return st.session_state["documents"].get(source_id, {})

def citation_bibliography() -> str:
    """Xuất danh mục tài liệu tham khảo theo đúng thứ tự bản nháp vừa tạo"""
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
# 12. PROMPT ENGINE – EVIDENCE ONLY (chuyên biệt cho Dược lâm sàng)
# ============================================================

BASE_SYSTEM_RULES = """
Bạn là trợ lý nghiên cứu khoa học, hỗ trợ viết luận văn Chuyên khoa cấp I
ngành Dược lâm sàng.

NGUYÊN TẮC BẮT BUỘC:
1. Tài liệu được cung cấp (PDF, tóm tắt PubMed, đoạn trích tạp chí Việt Nam)
   là nguồn bằng chứng ưu tiên duy nhất.
2. Không tự tạo số liệu, p-value, OR, RR, HR, CI95%, tỷ lệ %, liều dùng
   hoặc cỡ mẫu nếu không có trong bằng chứng.
3. Không tự tạo tên tác giả, năm, tên bài báo, DOI, PMID.
4. Không dùng kiến thức nền của mô hình làm bằng chứng nếu không có
   trong context được cung cấp.
5. Nếu context không đủ bằng chứng, phải nói rõ:
   "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận."
6. Mọi khẳng định dựa trên tài liệu phải chèn MÃ ĐỊNH DANH của tài liệu đó
   ngay sau câu. Ví dụ: "Tỷ lệ này là 12% [REF-001]."
7. TUYỆT ĐỐI KHÔNG tự tạo [1], [2], [3] và không dùng citation dạng tác giả-năm.
8. Phân biệt rõ: FACT (dữ kiện) – INTERPRETATION (diễn giải) – INFERENCE (suy luận).
9. Dùng chính xác thuật ngữ chuyên ngành Dược lâm sàng, văn phong khô khan.
"""

def generate_evidence_based(
    task: str, query: str, k: int = DEFAULT_TOP_K
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    
    # 1. Truy xuất bằng chứng (RAG)
    evidence = retrieve_evidence(query, k=k)
    if not evidence:
        return "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận.", [], []

    # 2. Đưa bằng chứng vào Citation Engine để lấy mã định danh [REF-...]
    engine = get_citation_engine()
    evidence_context = ""
    
    for ev in evidence:
        meta = st.session_state["documents"].get(ev["source_id"], {})
        # Đăng ký và lấy mã tag, VD: [REF-DOC_01]
        tag = engine.register_evidence(ev["source_id"], meta)
        
        table_note = f"\nGhi chú: {ev['table_hint']}" if ev.get("table_hint") else ""
        evidence_context += (
            f"\nTài liệu {tag}:\n"
            f"Nguồn: {ev['file_name']} | Trang: {ev['page']}\n"
            f"Nội dung: {ev['text']}{table_note}\n"
        )

    # 3. Gom Prompt và gọi Gemini
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

    # 4. Citation Engine xử lý hậu kỳ: Đổi [REF-...] thành [1], [2] theo thứ tự
    final_text, references, invalid_tags = engine.process_vancouver_citations(output)

    # Nếu AI tự bịa ra mã trích dẫn, thêm cảnh báo vào bài viết
    if invalid_tags:
        final_text += (
            f"\n\n> ⚠️ CẢNH BÁO AUDIT: Phát hiện AI tự tạo mã trích dẫn không có "
            f"trong dữ liệu truy xuất: {', '.join(invalid_tags)}. Đoạn này cần kiểm tra kỹ."
        )

    # 5. Lưu kết quả vào bộ nhớ
    st.session_state["last_generated"] = final_text
    st.session_state["last_evidence"] = evidence
    st.session_state["current_references"] = references  # Lưu để hàm citation_bibliography() đọc

    return final_text, evidence, invalid_tags
# ============================================================
# 17. KIỂM TRA NHẤT QUÁN SỐ LIỆU + TRÙNG LẶP NỘI BỘ (ĐÃ SỬA LỖI P0)
# ============================================================
import math

def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    # Cải tiến regex để bắt được số thập phân, phần trăm
    pattern = r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*%)?"
    return re.findall(pattern, text)

def parse_number(num_str: str) -> Optional[float]:
    """Hàm phụ trợ chuyển chuỗi số thành float để so sánh giá trị phái sinh (Level 2)"""
    clean_str = num_str.replace(",", ".").replace(" ", "").replace("%", "")
    try:
        return float(clean_str)
    except ValueError:
        return None

def compare_numbers_advanced(source_text: str, generated_text: str) -> Dict[str, Any]:
    """
    Audit số liệu 3 cấp độ:
    - Level 1: Exact match (Khớp chính xác bề mặt)
    - Level 2: Derived match (Khớp phái sinh, VD: 65% và 0.65)
    - Level 3: Warning (Cảnh báo số liệu lạ, không có trong Evidence)
    """
    source_nums = extract_numeric_tokens(source_text)
    generated_nums = extract_numeric_tokens(generated_text)

    source_normalized = set(x.replace(",", ".").replace(" ", "") for x in source_nums)
    generated_normalized = set(x.replace(",", ".").replace(" ", "") for x in generated_nums)
    
    # Chuyển đổi nguồn sang giá trị số học để kiểm tra Level 2
    source_floats = set(filter(None, [parse_number(x) for x in source_normalized]))

    exact_matches = []
    derived_matches = []
    warnings = []

    for gen_num in generated_normalized:
        # Bỏ qua các số trích dẫn kiểu [1], [2] (Citation Engine đã lo phần này)
        if re.match(r"^\[\d+\]$", gen_num):
            continue

        if gen_num in source_normalized:
            exact_matches.append(gen_num)
        else:
            # Kiểm tra Level 2: Có phải là dạng phái sinh (VD: 0.65 và 65%)
            gen_val = parse_number(gen_num)
            if gen_val is not None:
                is_derived = False
                for src_val in source_floats:
                    # Chấp nhận sai số nhỏ (rel_tol) và tỷ lệ x100 (tính phần trăm)
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

def audit_generated_text(text: str) -> Dict[str, Any]:
    """
    ĐÃ SỬA LỖI P0: Lấy đúng context của đoạn text cần audit thay vì dùng last_evidence cũ.
    """
    # 1. Truy xuất bằng chứng THỰC SỰ LIÊN QUAN đến đoạn văn đang được dán vào
    relevant_evidence = retrieve_evidence(text, k=6)
    source_text = "\n".join(e["text"] for e in relevant_evidence)
    
    # 2. Chạy Audit Số liệu 3 Cấp độ
    audit = compare_numbers_advanced(source_text, text)
    
    return {
        "evidence_used": relevant_evidence,
        **audit
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
# 18. XUẤT WORD (có phân tích heading/markdown cơ bản)
# ============================================================

def add_markdown_body_to_doc(doc: Document, body: str):
    """Chuyển đổi markdown đơn giản (###, ##, -, bảng) thành Word có định dạng."""
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
            # Bỏ qua dòng phân cách markdown table (---|---)
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
# 18.5. CÁC HÀM XỬ LÝ CHO TAB AUDIT (NGÔN NGỮ, ĐẠO VĂN, AI-STYLE)
# ============================================================

def spelling_and_terminology_check(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"""
    {BASE_SYSTEM_RULES}
    Bạn là một biên tập viên y khoa khó tính chuyên ngành Dược lâm sàng.
    Hãy rà soát đoạn văn bản sau để tìm ra:
    1. Các lỗi chính tả, lỗi đánh máy.
    2. Các lỗi dùng từ sai thuật ngữ chuyên ngành.

    ĐOẠN VĂN GỐC:
    {text}

    Chỉ trình bày những lỗi tìm thấy và đề xuất cách sửa. Nếu không có lỗi, hãy báo "✅ Không tìm thấy lỗi chính tả/thuật ngữ đáng kể".
    """
    # ÉP CHẠY LITE: Soi chính tả không cần suy luận sâu
    return call_gemini(prompt, model=MODEL_LITE)

def plagiarism_style_review(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"""
    {BASE_SYSTEM_RULES}
    Hãy đóng vai hội đồng phản biện luận văn CKI Dược lâm sàng.
    Phân tích đoạn văn sau để:
    1. Đánh giá độ logic và tính mạch lạc của các lập luận.
    2. Cảnh báo những câu văn có cấu trúc lặp lại quá nhiều (nguy cơ đạo văn cấu trúc).
    3. Đề xuất cách nâng cấp đoạn văn này thành văn phong học thuật, khô khan và trực diện hơn.

    ĐOẠN VĂN GỐC:
    {text}
    """
    return call_gemini(prompt)

def heuristic_ai_style_score(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"""
    {BASE_SYSTEM_RULES}
    Hãy phân tích đoạn văn sau và soi khắt khe các dấu hiệu nhận biết văn bản này có thể do AI (như ChatGPT/Gemini) viết:
    1. Việc lạm dụng các từ nối chuyển ý rập khuôn (Tóm lại, Có thể thấy rằng, Nhìn chung, Đáng chú ý là...).
    2. Cấu trúc câu quá máy móc, thiếu tính tự nhiên hoặc độ "gồ ghề" của văn phong do con người tự viết.
    3. Việc sử dụng tính từ hoa mỹ không phù hợp với văn bản khoa học chuyên khảo.

    ĐOẠN VĂN GỐC:
    {text}

    Trình bày dưới dạng gạch đầu dòng ngắn gọn các dấu hiệu "bốc mùi AI" tìm thấy và hướng dẫn người viết cách sửa lại cho tự nhiên.
    """
    # ÉP CHẠY LITE: Kiểm tra dấu hiệu AI là form mẫu cơ bản
    return call_gemini(prompt, model=MODEL_LITE)


# ============================================================
# 19. GIAO DIỆN
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
        "tìm trên PubMed, đồng thời tìm bài báo tiếng Việt liên quan. Kết quả "
        "chọn nạp sẽ được đưa vào cùng Evidence Database với PDF ở Tab 1, "
        "và được gắn SOURCE_TAG/citation như tài liệu gốc."
    )
    render_evidence_database_status()

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        t3_query = st.text_input(
            "Tên đề tài nghiên cứu (tiếng Việt):",
            placeholder=(
                "VD: Hiệu quả kiểm soát đường huyết bằng metformin ở bệnh nhân "
                "đái tháo đường type 2"
            ),
            key="t3_query_input",
        )
    with col_btn:
        max_res = st.number_input("Số bài/nguồn", min_value=2, max_value=10, value=5, key="t3_max_res")

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

            # --- PHẦN ĐÃ NÂNG CẤP DÀNH CHO TẠP CHÍ VN ---
            with st.spinner("Đang rút gọn từ khóa & tìm trên tạp chí Y học Việt Nam..."):
                vn_short_query = extract_vn_keywords(t3_query)
                st.session_state["t3_vn_keyword"] = vn_short_query # Lưu lại để hiện lên UI
                
                vn_results, vn_err = search_vn_journals(vn_short_query, max_res)
                st.session_state["t3_vn_data"] = vn_results
                if vn_err:
                    st.warning(vn_err)

    if st.session_state["t3_pm_data"] or st.session_state["t3_vn_data"]:
        st.write("---")
        col_vn, col_pm = st.columns(2)

        with col_vn:
            st.markdown("### 🇻🇳 Tạp chí Y học Việt Nam")
            # Hiện từ khóa VN đã rút gọn để anh biết AI đang tìm chữ gì
            if st.session_state.get("t3_vn_keyword"):
                st.success(f"🔑 Từ khoá đã rút gọn: **{st.session_state['t3_vn_keyword']}**")
                
            if not st.session_state["t3_vn_data"]:
                st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
            else:
                for i, art in enumerate(st.session_state["t3_vn_data"]):
                    with st.container(border=True):
                        st.markdown(f"**[{art['title']}]({art['link']})**" if art.get("link") else f"**{art['title']}**")
                        st.caption(art["source"])
                        st.write(art["snippet"])
                        if st.button("➕ Nạp vào Evidence Database", key=f"vn_ingest_{i}"):
                            added = ingest_vn_article(art)
                            if added:
                                # Update index incrementally cho RAG
                                rebuild_index(new_chunks=[added]) if isinstance(added, dict) else rebuild_index()
                                st.success("Đã nạp. Nhớ kiểm tra bản gốc trước khi dùng số liệu chi tiết.")
                            else:
                                st.info("Nguồn này đã có trong Evidence Database.")

        with col_pm:
            st.markdown("### 🌍 PubMed (Quốc tế)")
            if st.session_state["t3_en_keyword"]:
                st.success(f"🔑 Từ khoá MeSH: **{st.session_state['t3_en_keyword']}**")
            if not st.session_state["t3_pm_data"]:
                st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
            else:
                for i, art in enumerate(st.session_state["t3_pm_data"]):
                    with st.container(border=True):
                        st.markdown(f"**[{art['title']}]({art['url']})**")
                        st.caption(f"✍️ {art['authors']} ({art['year']}) — {art['journal']}")
                        with st.expander("Xem tóm tắt (Abstract)"):
                            st.write(art["abstract"])
                        if st.button("➕ Nạp vào Evidence Database", key=f"pm_ingest_{i}"):
                            added = ingest_pubmed_article(art)
                            if added:
                                rebuild_index()
                                st.success("Đã nạp vào Evidence Database.")
                            else:
                                st.info("Nguồn này đã có trong Evidence Database.")

        st.write("---")
        if st.button("➕ Nạp TẤT CẢ kết quả ở trên vào Evidence Database", key="t3_ingest_all"):
            count = 0
            for art in st.session_state["t3_pm_data"]:
                if ingest_pubmed_article(art):
                    count += 1
            for art in st.session_state["t3_vn_data"]:
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
    st.warning(
        "Đây là công cụ tạo bản nháp. Mọi citation và số liệu phải được kiểm "
        "tra lại (đối chiếu bản gốc) trước khi đưa vào luận văn chính thức."
    )
    render_evidence_database_status("dùng cho các nút viết nhanh bên dưới")

    my_research_data = st.text_area(
        "🌉 Số liệu nghiên cứu của riêng anh (dùng cho Bàn luận / So sánh):",
        placeholder=(
            "VD: 'Tỷ lệ nam/nữ là 1.42:1' hoặc dán bảng crosstab/kết quả "
            "logistic regression từ Tab 4 vào đây..."
        ),
        height=140,
    )

    citation_rules = """
QUY TẮC TRÍCH DẪN & HÀN LÂM BẮT BUỘC:
1. Chỉ dùng SOURCE_TAG thật để hệ thống tự chuyển thành [n].
2. Các số trích dẫn sẽ theo đúng thứ tự xuất hiện trong bài (do hệ thống cấp).
3. Không dùng kiểu [Tên tác giả, Năm].
4. Không dùng Heading 1 (#) hoặc Heading 2 (##) trong nội dung, chỉ dùng Heading 3 (###) cho tiêu đề mục.
"""

    # --- SỬA LỖI LAYOUT: TẠO CONTAINER FULL TRANG ---
    ket_qua_container = st.container()

    def run_quick_task(task_label: str, query: str, task_prompt: str, k: int):
        with st.spinner(f"AI đang soạn: {task_label}..."):
            output, evidence, invalid = generate_evidence_based(task_prompt, query, k=k)

        if output:
            with ket_qua_container:
                st.write("---")
                st.subheader(task_label)
                
                # In ra bản nháp luận văn
                st.markdown(output)

                # =========================================================
                # TÍNH NĂNG MỚI: EVIDENCE TRACE (DẤU VẾT BẰNG CHỨNG)
                # =========================================================
                st.markdown("### 🔎 Dấu vết bằng chứng (Evidence Trace)")
                st.caption("Bấm vào từng trích dẫn để xem chính xác đoạn văn bản gốc mà AI đã dùng làm căn cứ.")
                
                # Lấy danh sách các trích dẫn đã được sử dụng trong bản nháp này
                current_refs = st.session_state.get("current_references", [])
                
                if current_refs:
                    for ref in current_refs:
                        v_index = ref['vancouver_index']
                        ref_id = ref['ref_id']
                        # Bóc tách tiền tố REF- để lấy ID gốc của tài liệu
                        source_id = ref_id.replace("REF-", "") if ref_id.startswith("REF-") else ref_id
                        
                        # Lọc ra các đoạn văn bản (chunks) thuộc tài liệu này
                        related_chunks = [ev for ev in evidence if ev['source_id'] == source_id]
                        
                        title = ref['metadata'].get('title', 'Tài liệu chưa có tiêu đề')
                        file_name = ref['metadata'].get('file_name', 'N/A')
                        
                        # Tạo hộp Expander cho từng trích dẫn [1], [2]...
                        with st.expander(f"[{v_index}] ↳ {title[:85]}..."):
                            st.write(f"**Tệp gốc:** `{file_name}`")
                            if ref['metadata'].get('doi'):
                                st.write(f"**DOI:** {ref['metadata']['doi']}")
                                
                            # In ra các phân đoạn bằng chứng đã giúp AI viết câu
                            for chunk in related_chunks:
                                st.markdown(
                                    f"- **Trang/Mục:** `{chunk.get('page', 'N/A')}` | "
                                    f"**Độ khớp:** `{chunk.get('score', 0):.4f}` | "
                                    f"**Mã đoạn:** `{chunk.get('chunk_id', 'N/A')}`"
                                )
                                # Hiển thị text gốc trong hộp màu xanh
                                st.info(f"_{chunk.get('text', '')}_")
                else:
                    st.info("Đoạn văn này không sử dụng trích dẫn nào từ Evidence Database.")
                # =========================================================

                # Hiển thị Danh mục tài liệu tham khảo Vancouver chuẩn
                bib = citation_bibliography()
                with st.expander("📖 Danh mục Tài liệu tham khảo (Của bản nháp này)"):
                    st.code(bib if bib else "Chưa có citation registry.", language="text")

                # =========================================================
                # GIỮ NGUYÊN PHẦN AUDIT SỐ LIỆU TỰ ĐỘNG
                # =========================================================
                audit = audit_generated_text(output)
                colA, colB = st.columns(2)
                with colA:
                    if invalid:
                        st.error(f"Phát hiện citation ảo: {', '.join(invalid)}")
                    else:
                        st.success("Không phát hiện citation ảo.")
                with colB:
                    if audit.get("warnings"):
                        st.warning(f"Số liệu lạ (Cần kiểm tra lại): {', '.join(audit['warnings'])}")
                    else:
                        st.success("Không phát hiện số liệu lạ ngoài bằng chứng.")

                st.session_state["audit_log"].append({
                    "type": task_label, "invalid_citation": invalid, "audit": audit,
                })

    st.subheader("📝 Lệnh viết nhanh")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    # --- SỬA LỖI LAYOUT: CHỈ LƯU TRẠNG THÁI NÚT TRONG CỘT ---
    with c1: btn_dat_van_de = st.button("Đặt vấn đề")
    with c2: btn_tong_quan = st.button("Tổng quan tài liệu")
    with c3: btn_phuong_phap = st.button("Phương pháp NC")
    with c4: btn_ban_luan = st.button("Bàn luận toàn diện")
    with c5: btn_so_sanh = st.button("So sánh NC liên quan")
    with c6: btn_tltk = st.button("Trích dẫn TLTK")

    st.write("---")
    st.subheader("Lệnh tùy chỉnh")
    custom_prompt = st.text_area("Nhập câu lệnh khác:", key="custom_prompt_tab3")
    k_custom = st.slider("Số nguồn bằng chứng truy xuất", 3, MAX_TOP_K, DEFAULT_TOP_K, key="tk3")
    btn_custom = st.button("▶️ Chạy lệnh tùy chỉnh")

    # --- SỬA LỖI LAYOUT: THỰC THI HÀM BÊN NGOÀI CÁC CỘT HẸP ---
    if btn_dat_van_de:
        query = "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, dịch tễ học, gánh nặng bệnh tật liên quan sử dụng thuốc"
        task = f"""Viết phần 'Đặt vấn đề' cho luận văn CKI Dược lâm sàng.

YÊU CẦU BẮT BUỘC VỀ HÌNH THỨC:
1. Viết thành MỘT MẠCH VĂN LIỀN MẠCH, khoảng 400 từ, gồm 3-4 đoạn văn
   (paragraph) nối tiếp nhau theo trình tự logic: (i) bối cảnh chung/gánh
   nặng bệnh tật -> (ii) tính cấp thiết và nguyên tắc dược lâm sàng liên
   quan -> (iii) thực trạng sử dụng thuốc và khoảng trống bằng chứng hiện
   có -> (iv) câu dẫn vào lý do thực hiện nghiên cứu này.
2. TUYỆT ĐỐI KHÔNG dùng bất kỳ heading/tiêu đề phụ nào (không #, không
   ##, không ###, không in đậm dòng riêng làm tiêu đề mục con). Chỉ viết
   văn xuôi thuần túy, các đoạn cách nhau bằng một dòng trống.
3. Các đoạn phải kết nối mạch lạc với nhau bằng từ nối/ý chuyển tiếp tự
   nhiên (không lặp cấu trúc câu mở đầu giữa các đoạn), không viết rời
   rạc từng đoạn như các mục riêng biệt.
4. Văn phong khoa học Dược lâm sàng, khô khan, trực diện, không hoa mỹ,
   không liệt kê gạch đầu dòng.

{citation_rules}"""
        run_quick_task("Đặt vấn đề", query, task, k=6)

    if btn_tong_quan:
        query = "Tổng quan y văn, các nghiên cứu liên quan, cơ chế dược lý, kết quả chính, khuyến cáo điều trị"
        task = f"Viết phần 'Tổng quan tài liệu' chuyên sâu, tổng hợp các nghiên cứu và\nguideline liên quan, nêu bật cơ sở dược lý/lâm sàng.\n{citation_rules}"
        run_quick_task("Tổng quan tài liệu", query, task, k=8)

    if btn_phuong_phap:
        query = "Đối tượng nghiên cứu, tiêu chuẩn chọn loại, thiết kế nghiên cứu, cỡ mẫu, biến số nghiên cứu, phương pháp thu thập số liệu"
        task = f"Viết 'Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU'.\nMục biến số BẮT BUỘC trình bày dạng bảng Markdown 5 cột: TT | Tên biến | Định nghĩa | Phân loại | Kỹ thuật/công cụ thu thập.\n{citation_rules}"
        run_quick_task("Phương pháp nghiên cứu", query, task, k=5)

    if btn_ban_luan:
        if not my_research_data.strip():
            st.warning("Cần nhập số liệu của anh vào ô 'Số liệu nghiên cứu' trước!")
        else:
            task = f"KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:\n{my_research_data}\n\nYÊU CẦU TRÌNH BÀY:\n1. Chia Bàn luận thành các tiểu mục Heading 3 phù hợp với số liệu đã cho\n   (ví dụ ### 4.1. Đặc điểm bệnh nhân, ### 4.2. Thực trạng sử dụng thuốc,\n   ### 4.3. Kết quả điều trị và so sánh...).\n2. Mỗi tiểu mục viết thành đoạn văn hoàn chỉnh (không liệt kê gạch đầu dòng).\n3. Trong mỗi đoạn: nêu số liệu thực tế của tôi -> giải thích cơ chế dược\n   lý/lâm sàng -> đối chiếu trực tiếp với số liệu trong bằng chứng (cao hơn/\n   thấp hơn/tương đồng), có gắn SOURCE_TAG.\n4. Văn phong chuyên khảo Dược lâm sàng, không suy diễn ngoài dữ liệu.\n{citation_rules}"
            run_quick_task("Bàn luận toàn diện", my_research_data, task, k=8)

    if btn_so_sanh:
        if not my_research_data.strip():
            st.warning("Cần nhập số liệu của anh vào ô 'Số liệu nghiên cứu' trước!")
        else:
            task = f"KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:\n{my_research_data}\n\nTrình bày đúng cấu trúc (chỉ Heading 3):\n### 4.2. So sánh với các nghiên cứu và khuyến cáo\n(Lấy số liệu của tôi làm gốc, đối chiếu trực tiếp với bằng chứng, giải thích\nnguyên nhân khác biệt dựa trên cỡ mẫu/kỹ thuật/đặc thù đối tượng — chỉ khi\ncó căn cứ trong bằng chứng, nếu không có căn cứ phải ghi rõ đây là suy luận.)\n### 4.3. Ý nghĩa lâm sàng và thực tiễn\n### 4.4. Hạn chế của nghiên cứu\n{citation_rules}"
            run_quick_task("So sánh nghiên cứu liên quan", my_research_data, task, k=8)

    if btn_tltk:
        query = "Tài liệu tham khảo, tác giả, năm xuất bản, tạp chí"
        task = f"Chỉ liệt kê các SOURCE_TAG bạn thấy phù hợp là tài liệu tham khảo\nchính, không tự viết danh mục — danh mục chính thức lấy từ citation registry.\n{citation_rules}"
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

    # Khởi tạo giỏ chứa kết quả và kho lưu bảng thô trong bộ nhớ tạm
    if "result_cart" not in st.session_state:
        st.session_state["result_cart"] = []
    if "saved_tables" not in st.session_state:
        st.session_state["saved_tables"] = {}

    # ==========================================
    # BỘ DỌN RÁC TỰ ĐỘNG (AUTO-CLEAN)
    # ==========================================
    def auto_clean_data(raw_df: pd.DataFrame):
    logs = []
    df_clean = raw_df.copy()
    old_rows = df_clean.shape[0]
    
    # 1. Cắt khoảng trắng tên cột
    df_clean.columns = df_clean.columns.str.strip()
    
    # 2. Xóa cột rác Unnamed
    unnamed_cols = [c for c in df_clean.columns if "unnamed" in str(c).lower()]
    if unnamed_cols:
        df_clean = df_clean.drop(columns=unnamed_cols)
        logs.append(f"🗑️ Đã xóa {len(unnamed_cols)} cột rác (Unnamed) do phần mềm xuất dư.")
        
    # 3. Xóa dòng rỗng hoàn toàn
    df_clean = df_clean.dropna(how='all')
    if df_clean.shape[0] < old_rows:
        logs.append(f"🗑️ Đã xóa {old_rows - df_clean.shape[0]} dòng trống hoàn toàn.")
        
    # 4. TRÁNH TUYỆT ĐỐI việc biến đổi chữ hoa/thường (Không dùng capitalize) để bảo vệ mã ICD-10, HbA1c, eGFR.
    logs.append("🛡️ Đã giữ nguyên vẹn các giá trị gốc (không đổi hoa/thường) nhằm bảo vệ ký hiệu chuyên ngành (HbA1c, eGFR, ICD-10).")
        
    return df_clean, logs

    excel_file = st.file_uploader("Tải file Excel", type=["xlsx", "xls"], key="excel_data")

    if excel_file is not None:
        try:
            # 1. Đọc dữ liệu thô
            raw_df = pd.read_excel(excel_file)
            
            # 2. Chạy bộ dọn rác tự động
            with st.spinner("Đang dọn dẹp và chuẩn hóa dữ liệu..."):
                df, clean_logs = auto_clean_data(raw_df)
            
            st.success(f"Dữ liệu sẵn sàng: {df.shape[0]} dòng × {df.shape[1]} cột.")
            
            # 3. Hiển thị nhật ký dọn rác
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

            st.write("---")
            # --- HIỂN THỊ GIỎ KẾT QUẢ ---
            # --- HIỂN THỊ GIỎ KẾT QUẢ ---
            st.markdown(f"### 🛒 Giỏ kết quả: **{len(st.session_state['result_cart'])}** bảng đã lưu")
            st.info("💡 Mỗi khi anh bấm các nút thống kê bên dưới, kết quả sẽ tự động được nạp vào Giỏ này để lát nữa AI tuyển chọn.")
            if st.button("🗑️ Xóa toàn bộ Giỏ kết quả"):
                st.session_state["result_cart"] = []
                st.session_state["saved_tables"] = {}
                st.rerun()

            st.write("---")

            # ==========================================
            # 0. BỘ MÁY TUYỂN CHỌN & SẮP XẾP BẢNG
            # ==========================================
            st.subheader("📋 Bộ máy tuyển chọn & Sắp xếp bảng cho Chương Kết quả")
            st.info("Sau khi anh đã chạy các phép tính thống kê ở dưới để nạp dữ liệu vào Giỏ, hãy dùng công cụ này để lọc bảng đưa vào luận văn.")

            with st.expander("🎯 Khai báo Mục tiêu nghiên cứu (Gợi ý: Dùng chính xác tên cột trong Excel làm từ khóa)"):
                obj_input_1 = st.text_input("Mục tiêu 1", value="ĐẶC ĐIỂM BỆNH NHÂN NGHIÊN CỨU", key="obj_1")
                obj_input_2 = st.text_input("Mục tiêu 2", value="PHÂN TÍCH THỰC TRẠNG SỬ DỤNG THUỐC", key="obj_2")

                objectives = [
                    StudyObjective(id="MT1", title=obj_input_1, keywords=["tuổi", "tuoi", "giới", "gioi", "bệnh", "benh", "đặc điểm", "nhân khẩu", "bmi", "SoBHYT", "NgaySinh"]),
                    StudyObjective(id="MT2", title=obj_input_2, keywords=["thuốc", "thuoc", "phù hợp", "phu hop", "liều", "lieu", "chỉ định", "chi dinh", "hoạt chất", "icd", "TenHang"]),
                ]

            if st.button("🚀 Chạy Table Selection Engine & Lập mạch kể chuyện", type="primary", key="run_engine"):
                if not st.session_state["result_cart"]:
                    st.error("❌ Giỏ kết quả đang trống! Anh cần cuộn xuống dưới, bấm các nút 'Tính tần số', 'Quét Crosstab'... để nạp số liệu vào Giỏ trước.")
                else:
                    engine = TableSelectionEngine(objectives, st.session_state["result_cart"])
                    decisions = engine.run()
                    narrative_plan = NarrativePlanner.build(decisions)

                    st.session_state["selection_decisions"] = decisions
                    st.session_state["narrative_plan"] = narrative_plan
                    st.success("✅ Đã hoàn thành tuyển chọn, lọc trùng và sắp xếp cấu trúc Chương Kết quả!")

            if "selection_decisions" in st.session_state:
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
                st.info("Bôi đen các bảng dưới đây và bấm Ctrl+C để copy, sau đó sang Word bấm Ctrl+V để dán.")

                # Xuất toàn bộ bảng (kể cả OPTIONAL) ra định dạng HTML để copy sang Word giữ nguyên ô cột
                for d in st.session_state["selection_decisions"]:
                    if d.result_id in st.session_state["saved_tables"]:
                        st.markdown(f"**Bảng {d.recommended_order or '*'}. {d.title}** *(Xếp loại: {d.priority.value})*")

                        # Lấy dataframe từ bộ nhớ
                        df_table = st.session_state["saved_tables"][d.result_id]

                        # Chuyển thành bảng HTML chuẩn để copy dán Word không bị vỡ khung
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
            desc_vars = st.multiselect("Chọn biến phân loại", all_cols, key="desc_vars")

            if st.button("Tính tần số và tỷ lệ (Tự động nạp vào Giỏ)", key="calc_desc"):
                if not desc_vars:
                    st.warning("Vui lòng chọn ít nhất 1 biến.")
                else:
                    for var in desc_vars:
                        result = descriptive_table(df, var)
                        if not result.empty:
                            st.markdown(f"**► Biến: {var}**")
                            st.dataframe(result)

                            # NẠP VÀO GIỎ & LƯU BẢNG ĐỂ COPY
                            result_id = f"DESC_{var}"
                            st.session_state["saved_tables"][result_id] = result
                            st.session_state["result_cart"].append(
                                CandidateResult(
                                    id=result_id,
                                    title=f"Đặc điểm phân bố của biến {var}",
                                    result_type="demographic",
                                    variables=[var],
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
                num_vars = st.multiselect("Chọn biến định lượng", numeric_candidates, key="num_vars")
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

                                # NẠP VÀO GIỎ & LƯU BẢNG ĐỂ COPY
                                result_id = f"NUM_{var}"
                                st.session_state["saved_tables"][result_id] = num_df
                                st.session_state["result_cart"].append(
                                    CandidateResult(
                                        id=result_id,
                                        title=f"Đặc điểm định lượng của biến {var}",
                                        result_type="baseline",
                                        variables=[var],
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
                if st.checkbox("✅ Chọn tất cả biến phụ thuộc", key="chk_all_deps"):
                    deps = st.multiselect("Các biến phụ thuộc", all_cols, default=all_cols, key="cross_deps")
                else:
                    deps = st.multiselect("Các biến phụ thuộc", all_cols, key="cross_deps")
            with cc2:
                if st.checkbox("✅ Chọn tất cả biến độc lập", key="chk_all_indeps"):
                    indeps = st.multiselect("Các biến độc lập cần đối chiếu", all_cols, default=all_cols, key="cross_indeps")
                else:
                    indeps = st.multiselect("Các biến độc lập cần đối chiếu", all_cols, key="cross_indeps")

            if st.button("Quét Crosstab + Kiểm định (Nạp TẤT CẢ vào Giỏ)", key="calc_cross"):
                if not deps or not indeps:
                    st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
                else:
                    found_count = 0
                    processed_pairs = set()
                    with st.spinner("Đang cày xới toàn bộ ma trận số liệu..."):
                        for dep in deps:
                            for indep in indeps:
                                if dep == indep: continue
                                pair_key = frozenset([dep, indep])
                                if pair_key in processed_pairs: continue
                                processed_pairs.add(pair_key)

                                try:
                                    result = crosstab_test(df, indep, dep)
                                    found_count += 1
                                    
                                    # CHỐNG P-HACKING: Không lọc p < 0.05 nữa, đánh dấu bằng màu sắc
                                    sig_marker = "🟢 CÓ Ý NGHĨA" if result['p_value'] < 0.05 else "⚪ KHÔNG Ý NGHĨA"
                                    st.markdown(f"**► Mối liên quan giữa: [{indep}] & [{dep}] — {sig_marker}**")
                                    st.dataframe(result["table"])
                                    st.write(f"- **Kiểm định:** {result['test']} | **p-value:** `{result['p_value']:.4g}`")
                                    
                                    # HIỆN EFFECT SIZE NẾU CÓ
                                    if "effect_size" in result:
                                        st.write(f"- **Chỉ số (Effect Size):** `{result['effect_size']}`")

                                    # NẠP VÀO GIỎ TẤT CẢ KẾT QUẢ
                                    result_id = f"CROSS_{indep}_{dep}"
                                    st.session_state["saved_tables"][result_id] = result["table"]
                                    st.session_state["result_cart"].append(
                                        CandidateResult(
                                            id=result_id,
                                            title=f"Mối liên quan giữa {indep} và {dep}",
                                            result_type="association",
                                            variables=[indep, dep],
                                            p_value=result['p_value'],
                                            scientific_value=4.5, clinical_importance=4.5, discussion_value=5.0
                                        )
                                    )
                                except Exception as e:
                                    # HIỆN RÕ LỖI THAY VÌ PASS
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
                if st.checkbox("✅ Chọn tất cả biến nhóm", key="chk_all_groups"):
                    group_vars = st.multiselect("Biến nhóm (Tự lọc biến 2 mức)", all_cols, default=all_cols, key="group_vars")
                else:
                    group_vars = st.multiselect("Biến nhóm (Tự lọc biến 2 mức)", all_cols, key="group_vars")
            with gc2:
                valid_num_cols = numeric_candidates if 'numeric_candidates' in locals() and numeric_candidates else all_cols
                if st.checkbox("✅ Chọn tất cả biến định lượng", key="chk_all_vals"):
                    val_vars = st.multiselect("Biến định lượng cần so sánh", valid_num_cols, default=valid_num_cols, key="val_vars")
                else:
                    val_vars = st.multiselect("Biến định lượng cần so sánh", valid_num_cols, key="val_vars")

            if st.button("Quét kiểm định so sánh (Nạp TẤT CẢ vào Giỏ)", key="run_group_compare"):
                if not group_vars or not val_vars:
                    st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
                else:
                    found_count = 0
                    with st.spinner("Đang rà soát và tính toán..."):
                        for gv in group_vars:
                            for vv in val_vars:
                                if gv == vv: continue
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
                                            id=result_id,
                                            title=f"Sự khác biệt của biến {vv} giữa các nhóm {gv}",
                                            result_type="association",
                                            variables=[gv, vv],
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
            # 5. HỒI QUY LOGISTIC NHỊ PHÂN (SMART SELECT)
            # ==========================================
            st.subheader("5. Hồi quy logistic nhị phân (OR và 95% CI)")
            
            # 1. LỌC THÔNG MINH BIẾN KẾT CỤC (Chỉ lấy biến có đúng 2 giá trị)
            outcome_candidates = [c for c in all_cols if df[c].dropna().nunique() == 2]
            
            # 2. LỌC THÔNG MINH YẾU TỐ DỰ BÁO (Bỏ cột Unnamed, ID, Tên, Ngày tháng, biến chỉ có 1 giá trị)
            forbidden_keywords = ["unnamed", "ngay", "ngày", "ten", "tên", "ma", "mã", "sobenhan", "id"]
            predictor_candidates = [
                c for c in all_cols 
                if not any(kw in str(c).lower() for kw in forbidden_keywords)
                and df[c].dropna().nunique() > 1
            ]

            lc1, lc2 = st.columns([1, 2])
            with lc1:
                if st.checkbox("✅ Chọn tất cả biến kết cục", key="chk_all_outcomes"):
                    outcomes = st.multiselect("Biến kết cục (Đã tự động lọc chuẩn Nhị phân)", outcome_candidates, default=outcome_candidates, key="log_outcomes")
                else:
                    outcomes = st.multiselect("Biến kết cục (Đã tự động lọc chuẩn Nhị phân)", outcome_candidates, key="log_outcomes")
            with lc2:
                if st.checkbox("✅ Chọn tất cả yếu tố dự báo", key="chk_all_predictors"):
                    predictors = st.multiselect("Yếu tố dự báo (Đã tự động bỏ ID, Ngày, Cột rác)", predictor_candidates, default=predictor_candidates, key="log_predictors")
                else:
                    predictors = st.multiselect("Yếu tố dự báo (Đã tự động bỏ ID, Ngày, Cột rác)", predictor_candidates, key="log_predictors")

            if st.button("Chạy Logistic Regression đa biến (Nạp vào Giỏ)", key="run_logistic"):
                if not outcomes or not predictors:
                    st.warning("Chọn ít nhất một biến ở mỗi bên.")
                else:
                    found_count = 0
                    with st.spinner("Đang xây dựng mô hình hồi quy..."):
                        for out in outcomes:
                            preds = [p for p in predictors if p != out]
                            if not preds: continue
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
                                        id=result_id,
                                        title=f"Mô hình hồi quy logistic đánh giá yếu tố liên quan đến {out}",
                                        result_type="regression",
                                        variables=[out] + preds,
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
            interpretation_request = st.text_area("Dán bảng kết quả thô vào đây", height=160, key="interpretation_request")

            if st.button("AI diễn giải", key="ai_interpret"):
                if not interpretation_request.strip():
                    st.warning("Nhập kết quả trước.")
                else:
                    prompt = f"""{BASE_SYSTEM_RULES}\nBạn chỉ được DIỄN GIẢI kết quả thống kê dưới đây. Không được tính lại hoặc sửa số liệu.\nKẾT QUẢ:\n{interpretation_request}"""
                    response = call_gemini(prompt)
                    if response:
                        st.markdown(response)

        except Exception as exc:
            st.error(f"Lỗi đọc file Excel: {exc}")

# ------------------------------------------------------------
# TAB 5 – AUDIT (1 HÀNG GỌN GÀNG - 6 CHỨC NĂNG)
# ------------------------------------------------------------
with tabs[4]:
    st.header("🔎 Audit luận văn toàn diện")

    st.markdown(
        '<div class="warning-box">⚠️ <b>Giới hạn cần biết:</b> Các công cụ ở '
        "tab này chỉ đưa ra <b>chỉ báo nguy cơ / gợi ý kiểm tra thêm</b>. "
        "Không có công cụ nào khẳng định chắc chắn 100%.</div>",
        unsafe_allow_html=True,
    )

    audit_text = st.text_area("Dán đoạn văn cần kiểm tra vào đây:", height=250, key="audit_text")
    
    # Tạo 1 hàng duy nhất gồm 6 cột bằng nhau
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    st.write("---")
    # Khung hiển thị kết quả nằm cố định bên dưới để không làm nhảy giao diện các nút
    ket_qua_audit_container = st.container()

    # 1. AUDIT SỐ LIỆU
    with c1:
        if st.button("🔢 Số liệu", key="audit_numbers", use_container_width=True):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang truy xuất và đối chiếu..."):
                    result = audit_generated_text(audit_text)
                with ket_qua_audit_container:
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

    # 2. AUDIT TRÍCH DẪN
    with c2:
        if st.button("📚 Trích dẫn", key="audit_citation", use_container_width=True):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                citations_in_text = re.findall(r"\[(\d+)\]", audit_text)
                current_refs = {str(ref['vancouver_index']): ref for ref in st.session_state.get("current_references", [])}
                with ket_qua_audit_container:
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
        if st.button("🔍 Trùng lặp", key="audit_overlap", use_container_width=True):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang quét Jaccard Similarity..."):
                    overlaps = internal_overlap_audit(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🔍 Kết quả tìm trùng lặp nội bộ")
                    if not overlaps:
                        st.info("Không tìm thấy đoạn trùng đáng kể trong kho tài liệu hiện tại.")
                    else:
                        for item in overlaps:
                            st.markdown(f"**{item['file']} – trang {item['page']}**\nSimilarity: **{item['similarity']}**\n\n> {item['text']}")

    # 4. KIỂM TRA CHÍNH TẢ
    with c4:
        if st.button("🔤 Chính tả", key="audit_spelling", use_container_width=True):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang rà soát lỗi chính tả & thuật ngữ..."):
                    response = spelling_and_terminology_check(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🔤 Kiểm tra Chính tả & Thuật ngữ")
                    st.markdown(response if response else "Không nhận được phản hồi từ AI.")

    # 5. KIỂM TRA VĂN PHONG AI
    with c5:
        if st.button("🤖 Check văn AI", key="audit_ai_style", use_container_width=True):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang phân tích dấu hiệu văn phong máy móc..."):
                    style_analysis = heuristic_ai_style_score(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🤖 Chỉ báo nguy cơ văn bản do AI viết")
                    st.markdown(style_analysis if style_analysis else "Không nhận được phản hồi từ AI.")

    # 6. PHẢN BIỆN LOGIC
    with c6:
        if st.button("⚖️ Phản biện", key="logic_review", use_container_width=True):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang soi logic nghiên cứu..."):
                    prompt = f"""
                    {BASE_SYSTEM_RULES}
                    Đóng vai phản biện luận văn CKI Dược lâm sàng. Đọc đoạn văn sau và chỉ ra các điểm yếu logic:
                    1. Có khẳng định nào thiếu bằng chứng không?
                    2. Có nhảy từ tương quan sang nhân quả không?
                    3. Kết luận có vượt quá giới hạn thiết kế nghiên cứu không?
                    ĐOẠN VĂN GỐC: {audit_text}
                    """
                    response = call_gemini(prompt)
                with ket_qua_audit_container:
                    st.markdown("### ⚖️ Kết quả Phản biện Logic")
                    st.markdown(response if response else "Không nhận được phản hồi từ AI.")
# ------------------------------------------------------------
# TAB 6 – NGUỒN & CẤU HÌNH
# ------------------------------------------------------------
with tabs[5]:
    st.header("⚙️ Nguồn, citation và cấu hình")

    st.write(f"**Gemini model:** `{DEFAULT_MODEL}`")
    st.write(f"**Embedding model:** `{DEFAULT_EMBEDDING}`")
    st.write(f"**SerpAPI (tra cứu tạp chí VN):** {'✅ đã cấu hình' if get_serpapi_key() else '❌ chưa cấu hình (tùy chọn)'}")

    st.write("---")
    st.subheader("Danh sách domain tạp chí Y học Việt Nam (dùng cho Tab 2)")

    domains_text = st.text_area(
        "Mỗi domain một dòng:",
        value="\n".join(st.session_state["vn_journal_domains"]),
        height=120,
    )
    if st.button("💾 Lưu danh sách domain"):
        st.session_state["vn_journal_domains"] = [
            d.strip() for d in domains_text.splitlines() if d.strip()
        ]
        st.success("Đã cập nhật.")

    st.write("---")
    st.subheader("Citation Registry")

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
    st.subheader("Metadata nguồn (bổ sung tay - AI không tự điền)")

    for source_id, meta in list(st.session_state["documents"].items()):
        with st.expander(f"[{meta.get('origin','')}] {source_id} – {meta['file_name']}"):
            mc1, mc2 = st.columns(2)
            with mc1:
                authors = st.text_input("Tác giả", value=meta.get("authors", ""), key=f"authors_{source_id}")
                title = st.text_input("Tên bài/tài liệu", value=meta.get("title", ""), key=f"title_{source_id}")
                year = st.text_input("Năm", value=meta.get("year", ""), key=f"year_{source_id}")
            with mc2:
                journal = st.text_input("Tạp chí", value=meta.get("journal", ""), key=f"journal_{source_id}")
                doi = st.text_input("DOI", value=meta.get("doi", ""), key=f"doi_{source_id}")
                pmid = st.text_input("PMID", value=meta.get("pmid", ""), key=f"pmid_{source_id}")

            if st.button("💾 Lưu metadata", key=f"save_{source_id}"):
                meta.update({
                    "authors": authors, "title": title, "year": year,
                    "journal": journal, "doi": doi, "pmid": pmid,
                })
                st.session_state["documents"][source_id] = meta
                st.success("Đã lưu metadata.")

    st.write("---")
    st.subheader("Nguyên tắc sử dụng")
    st.markdown(
        """
- Không xem bản nháp AI là kết quả cuối cùng.
- Không dùng citation nếu chưa truy ngược được về nguồn.
- Đoạn trích từ tạp chí Việt Nam (Google Scholar) chỉ là snippet ngắn —
  luôn đối chiếu bản gốc trước khi dùng số liệu chi tiết.
- Không để AI tính p-value, OR, CI95% hoặc tỷ lệ khi Python có thể tính trực tiếp.
- Không suy luận quan hệ nhân quả từ nghiên cứu quan sát nếu thiết kế không cho phép.
- Không gọi chức năng audit nội bộ là "chứng nhận không đạo văn".
- Không có công cụ nào bảo đảm tuyệt đối văn bản "không phải AI viết".
- Người nghiên cứu phải kiểm tra bản gốc trước khi chấp nhận số liệu và diễn giải.
"""
    )

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

    # ============================================================
    # QUẢN LÝ CHECKPOINT & DỰ ÁN LUẬN VĂN (LƯU/TẢI XUỐNG ĐĨA)
    # ============================================================
    st.write("---")
    st.subheader("💾 Quản lý Checkpoint & Dự án luận văn")
    st.info(
        "Lưu lại toàn bộ trạng thái làm việc hiện tại (Evidence Database, "
        "Citation Registry, Giỏ kết quả & cấu trúc Chương 3 ở Tab 4, bản "
        "nháp gần nhất...) xuống một checkpoint trên máy chủ, để có thể "
        "khôi phục lại khi cần — ví dụ sau khi app bị restart hoặc khi "
        "muốn chuyển sang làm đề tài khác rồi quay lại sau."
    )

    col_save1, col_save2 = st.columns([3, 1])
    with col_save1:
        proj_name_input = st.text_input(
            "Tên định danh đề tài / dự án:",
            placeholder="VD: Vancomycin_SonLa_2026 hoặc THA_BVTP_2026",
            key="proj_name_input",
        )
    with col_save2:
        st.write("")
        st.write("")
        if st.button("💾 Lưu Checkpoint", type="primary", key="btn_save_project"):
            if not proj_name_input.strip():
                st.warning("Vui lòng nhập tên dự án trước khi lưu.")
            else:
                success, msg = save_project(proj_name_input)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    st.write("📂 **Danh sách checkpoint đã lưu trên máy chủ:**")
    existing_projects = list_projects()
    if existing_projects:
        col_load1, col_load2, col_load3 = st.columns([3, 1, 1])
        with col_load1:
            selected_proj = st.selectbox(
                "Chọn checkpoint để khôi phục:", existing_projects, key="select_proj_load"
            )
        with col_load2:
            if st.button("📂 Tải lại", key="btn_load_project"):
                success, msg = load_project(selected_proj)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with col_load3:
            if st.button("🗑️ Xóa", key="btn_delete_project"):
                success, msg = delete_project(selected_proj)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.caption("Chưa có checkpoint nào được lưu.")

    st.caption(
        "⚠️ Lưu ý: Checkpoint được lưu cục bộ trên máy chủ đang chạy app. "
        "Nếu deploy trên Streamlit Community Cloud, checkpoint có thể mất khi "
        "app 'ngủ đông' / redeploy — chỉ nên xem đây là điểm lưu tạm trong "
        "phiên làm việc, luôn xuất Word (nút ở trên) để lưu bản nháp quan trọng."
    )
