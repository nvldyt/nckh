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
    # ÉP CHẠY LITE: Tác vụ dịch thuật đơn giản, tiết kiệm quota
    text = call_gemini(prompt, model=MODEL_LITE, temperature=0.1)
    if text:
        return text.strip().strip('"').strip("'")
    return vietnamese_query


# ============================================================
# 10. INDEX / VECTOR RETRIEVAL (dùng chung cho PDF + kết quả tra cứu)
# ============================================================
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
                                rebuild_index()
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
            # --- SỬA LỖI LAYOUT: ĐẨY KẾT QUẢ VÀO CONTAINER Ở TRÊN ---
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
                        st.error("Có citation không hợp lệ trong bản nháp.")
                    else:
                        st.success("Không phát hiện citation invalid.")
                with colB:
                    if audit["suspicious_generated_numbers"]:
                        st.warning(f"Số liệu lạ (không có trong bằng chứng): {audit['suspicious_generated_numbers']}")
                    else:
                        st.success("Không phát hiện số liệu lạ ngoài bằng chứng đã truy xuất.")

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

    excel_file = st.file_uploader("Tải file Excel", type=["xlsx", "xls"], key="excel_data")

    if excel_file is not None:
        try:
            df = pd.read_excel(excel_file)
            st.success(f"Dữ liệu: {df.shape[0]} dòng × {df.shape[1]} cột.")

            validation = validate_dataframe(df)
            if validation:
                for item in validation:
                    st.warning(item)

            with st.expander("Xem dữ liệu thô"):
                st.dataframe(df)

            st.write("---")
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
            st.subheader("3. Bảng chéo và kiểm định (Chi-square / Fisher)")
            cc1, cc2 = st.columns(2)
            with cc1:
                deps = st.multiselect("Các biến phụ thuộc", all_cols, key="cross_deps")
            with cc2:
                indeps = st.multiselect("Các biến độc lập cần đối chiếu", all_cols, key="cross_indeps")

            if st.button("Quét Crosstab + Kiểm định (Nạp các biến có p < 0.05 vào Giỏ)", key="calc_cross"):
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
                                    if result['p_value'] is not None and result['p_value'] < 0.05:
                                        found_count += 1
                                        st.markdown(f"**► Mối liên quan CÓ Ý NGHĨA giữa: [{indep}] & [{dep}]**")
                                        st.dataframe(result["table"])
                                        st.write(f"- **Kiểm định:** {result['test']} | **p-value:** `{result['p_value']:.6g}` 🟢")

                                        # NẠP VÀO GIỎ & LƯU BẢNG ĐỂ COPY
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
                                except Exception:
                                    pass

                    if found_count > 0:
                        st.success(f"✅ Đã tìm thấy và nạp {found_count} bảng kiểm định vào Giỏ!")
                    else:
                        st.info("Không tìm thấy cặp biến nào có p < 0.05.")

            st.write("---")

            # ==========================================
            # 4. SO SÁNH 2 NHÓM
            # ==========================================
            st.subheader("4. So sánh biến định lượng giữa 2 nhóm (T-test / Mann-Whitney)")
            gc1, gc2 = st.columns(2)
            with gc1:
                group_vars = st.multiselect("Biến nhóm (Tự lọc biến 2 mức)", all_cols, key="group_vars")
            with gc2:
                val_vars = st.multiselect("Biến định lượng cần so sánh", numeric_candidates or all_cols, key="val_vars")

            if st.button("Quét kiểm định so sánh hàng loạt (Nạp vào Giỏ)", key="run_group_compare"):
                if not group_vars or not val_vars:
                    st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
                else:
                    found_count = 0
                    with st.spinner("Đang rà soát và so sánh các nhóm..."):
                        for gv in group_vars:
                            for vv in val_vars:
                                if gv == vv: continue
                                try:
                                    result = compare_two_groups(df, gv, vv)
                                    if result['p_value'] is not None and result['p_value'] < 0.05:
                                        found_count += 1
                                        g1n, g2n = result["group_names"]
                                        st.markdown(f"**► Sự khác biệt CÓ Ý NGHĨA của [{vv}] giữa 2 nhóm [{gv}]**")
                                        st.write(f"**p-value:** `{result['p_value']:.6g}` 🟢")
                                        comp_df = pd.DataFrame({
                                            g1n: result["group1_stats"],
                                            g2n: result["group2_stats"],
                                        })

                                        # NẠP VÀO GIỎ & LƯU BẢNG ĐỂ COPY
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
                                except Exception:
                                    pass
                    if found_count > 0:
                        st.success(f"✅ Đã nạp {found_count} kết quả so sánh vào Giỏ!")
                    else:
                        st.info("Không có sự khác biệt (p < 0.05).")

            st.write("---")

            # ==========================================
            # 5. HỒI QUY LOGISTIC NHỊ PHÂN
            # ==========================================
            st.subheader("5. Hồi quy logistic nhị phân (Tìm yếu tố nguy cơ độc lập)")
            lc1, lc2 = st.columns([1, 2])
            with lc1:
                outcomes = st.multiselect("Biến kết cục (Tự lọc biến nhị phân)", all_cols, key="log_outcomes")
            with lc2:
                predictors = st.multiselect("Các yếu tố đưa vào mô hình", all_cols, key="log_predictors")

            if st.button("Quét Logistic Regression hàng loạt (Nạp vào Giỏ)", key="run_logistic"):
                if not outcomes or not predictors:
                    st.warning("Chọn ít nhất một biến ở mỗi bên.")
                else:
                    found_count = 0
                    with st.spinner("Đang chạy hàng loạt mô hình hồi quy đa biến..."):
                        for out in outcomes:
                            preds = [p for p in predictors if p != out]
                            if not preds: continue
                            try:
                                result_df, summary = binary_logistic_regression(df, out, preds)
                                sig_df = result_df[result_df['p-value'] < 0.05]

                                if not sig_df.empty:
                                    found_count += 1
                                    st.markdown(f"**► CÁC YẾU TỐ ĐỘC LẬP có tác động tới kết cục: [{out}]**")
                                    st.dataframe(sig_df)

                                    # NẠP VÀO GIỎ & LƯU BẢNG ĐỂ COPY
                                    result_id = f"LOG_{out}"
                                    st.session_state["saved_tables"][result_id] = sig_df
                                    st.session_state["result_cart"].append(
                                        CandidateResult(
                                            id=result_id,
                                            title=f"Mô hình hồi quy đa biến cho kết cục: {out}",
                                            result_type="regression",
                                            variables=[out] + preds,
                                            scientific_value=5.0, clinical_importance=5.0, discussion_value=5.0
                                        )
                                    )
                            except Exception:
                                pass

                    if found_count > 0:
                        st.success(f"✅ Đã nạp {found_count} mô hình hồi quy vào Giỏ!")
                    else:
                        st.info("Không tìm thấy mô hình có ý nghĩa thống kê (p < 0.05).")

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
# TAB 5 – AUDIT
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
                    st.write("**Số xuất hiện trong nguồn (bằng chứng đã truy xuất gần nhất):**")
                    st.write(result["source_numbers"])
                    st.write("**Số xuất hiện trong bản viết:**")
                    st.write(result["generated_numbers"])
                    if result["suspicious_generated_numbers"]:
                        st.error("Có số cần kiểm tra:")
                        st.write(result["suspicious_generated_numbers"])
                    else:
                        st.success("Không phát hiện số mới ngoài tập bằng chứng đang truy xuất.")

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
                        st.success("Không phát hiện citation invalid theo bộ kiểm tra hiện tại.")

    with a3:
        if st.button("🔍 Tìm trùng lặp nội bộ", key="audit_overlap"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                overlaps = internal_overlap_audit(audit_text)
                with ket_qua_audit_container:
                    st.markdown("### 🔍 Kết quả tìm trùng lặp nội bộ")
                    if not overlaps:
                        st.info("Không tìm thấy đoạn trùng đáng kể trong kho tài liệu hiện tại.")
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
                    if response:
                        st.markdown(response)
                    else:
                        st.error("Không nhận được kết quả từ AI.")

    with b2:
        if st.button("📄 Kiểm tra nguy cơ đạo văn (mở rộng)", key="audit_plagiarism"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                # --- SỬA LỖI: tính overlaps ngay tại đây, không dựa vào biến
                # còn sót lại từ lần bấm nút "Tìm trùng lặp nội bộ" (a3) ở trên,
                # vì mỗi lần Streamlit rerun lại toàn bộ script, "overlaps" sẽ
                # không tồn tại nếu người dùng chưa từng bấm nút đó trước.
                with st.spinner("Đang đối chiếu n-gram và phân tích diễn đạt..."):
                    overlaps = internal_overlap_audit(audit_text)
                    response = plagiarism_style_review(audit_text)

                with ket_qua_audit_container:
                    st.markdown("### 📄 Kết quả kiểm tra Nguy cơ đạo văn")
                    st.caption(
                        "Phạm vi đối chiếu: chỉ các nguồn đã nạp trong Evidence "
                        "Database (Tab 1 + Tab 2) của phiên hiện tại."
                    )

                    if overlaps:
                        max_sim = max(o["similarity"] for o in overlaps)
                        if max_sim >= 0.3:
                            st.error(f"Tỷ lệ trùng n-gram cao nhất: {max_sim*100:.1f}% — cần xem lại diễn đạt.")
                        elif max_sim >= 0.1:
                            st.warning(f"Tỷ lệ trùng n-gram cao nhất: {max_sim*100:.1f}% — nên kiểm tra thêm.")
                        else:
                            st.info(f"Tỷ lệ trùng n-gram cao nhất: {max_sim*100:.1f}% — khá thấp.")

                        with st.expander("Xem chi tiết các đoạn trùng n-gram cao nhất"):
                            for item in overlaps:
                                st.markdown(
                                    f"""**{item['file']} – trang {item['page']}** — Similarity: **{item['similarity']}**

> {item['text']}
"""
                                )
                    else:
                        st.info("Không phát hiện trùng cụm từ dài với nguồn đã nạp.")

                    if response:
                        st.markdown(response)
                    else:
                        st.error("Không nhận được nhận xét từ AI.")

    with b3:
        if st.button("🤖 Chỉ báo nguy cơ văn bản do AI viết", key="audit_ai_style"):
            if not audit_text.strip():
                st.warning("Chưa có văn bản.")
            else:
                with st.spinner("Đang phân tích phong cách văn bản..."):
                    style_analysis = heuristic_ai_style_score(audit_text)

                with ket_qua_audit_container:
                    st.markdown("### 🤖 Chỉ báo nguy cơ văn bản do AI viết")
                    st.caption(
                        "Đây là đánh giá dựa trên phong cách ngôn ngữ, mang tính tham "
                        "khảo — KHÔNG phải kết luận chắc chắn 100% văn bản do AI viết hay không."
                    )

                    if style_analysis:
                        st.markdown(style_analysis)
                    else:
                        st.error("Không nhận được kết quả phân tích từ AI.")

    st.write("---")
    st.subheader("Phản biện logic bằng AI")

    logic_request = st.text_area("Mô tả vấn đề hoặc dán đoạn văn", height=180, key="logic_request")

    if st.button("⚖️ Phản biện logic", key="logic_review"):
        if not logic_request.strip():
            st.warning("Nhập nội dung cần phản biện.")
        else:
            with st.spinner("AI đang xử lý..."):
                prompt = f"""
{BASE_SYSTEM_RULES}
Đóng vai phản biện luận văn CKI Dược lâm sàng.

NỘI DUNG:
{logic_request}

Hãy kiểm tra:
1. Có khẳng định nào không có bằng chứng?
2. Có nhảy logic từ tương quan sang nhân quả không?
3. Có số liệu nào không có nguồn?
4. Có kết luận vượt quá thiết kế nghiên cứu không?
5. Có khái niệm dược lý/lâm sàng nào bị dùng sai không?
6. Có chỗ nào cần bổ sung bằng chứng?
7. Có câu nào nên viết thận trọng hơn?

Không được tự bổ sung tài liệu hoặc số liệu.
"""
                response = call_gemini(prompt)
                if response:
                    st.markdown(response)

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
