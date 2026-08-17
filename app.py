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

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from docx import Document

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
# 1. CẤU HÌNH & GIAO DIỆN KHOA HỌC (ACADEMIC CLEAN THEME)
# ============================================================

st.set_page_config(
    page_title="NCKH - Dược lâm sàng",
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

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li, .stMarkdown {
        font-family: 'Be Vietnam Pro', 'Arial', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .stApp {
        background-color: #f4f6f9;
    }

    .block-container { 
        max-width: 1450px; 
        padding-top: 3.5rem !important; 
        padding-bottom: 2rem !important;
    }

    h1 {
        color: #1e293b !important;
        text-align: center;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        text-align: center;
        color: #64748b !important;
    }
    h2, h3 { color: #1e3a8a !important; font-weight: 600; }

    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #e2e8f0;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a8a !important;
        color: #fff !important;
    }

    div.stButton > button, div.stDownloadButton > button {
        background: #1e3a8a !important;
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #1d4ed8 !important;
    }

    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }

    .warning-box { border-left: 4px solid #f59e0b; padding: 10px 14px; background: #fffbeb; border-radius: 6px; color: #92400e; }
    .danger-box  { border-left: 4px solid #ef4444; padding: 10px 14px; background: #fef2f2; border-radius: 6px; color: #991b1b; }
    .success-box { border-left: 4px solid #10b981; padding: 10px 14px; background: #ecfdf5; border-radius: 6px; color: #065f46; }

    .stMarkdown p, .stMarkdown li {
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        text-align: justify !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 2. SESSION STATE
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
# 3. GEMINI CLIENT & EMBEDDING
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
        st.error("Chưa có GEMINI_API_KEY trong Streamlit Secrets hoặc biến môi trường.")
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
                    status = st.warning(f"⏳ Máy chủ Google đang quá tải. Tự động đợi {wait_time}s rồi thử lại (Lần {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    status.empty()
                else:
                    st.error("❌ Máy chủ Google Gemini đang quá bận. Vui lòng thử lại sau ít phút!")
                    return None
            else:
                if attempt == max_retries - 1:
                    st.error(f"Lỗi Gemini: {error_msg}")
                    return None
                time.sleep(3)
    return None

@st.cache_resource
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

def get_embeddings(texts: List[str]) -> np.ndarray:
    model = load_embedding_model(DEFAULT_EMBEDDING)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)

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
            '"chưa đủ bằng chứng để kết luận". Hãy nạp PDF ở Tab 1 hoặc tra cứu trước.</div>',
            unsafe_allow_html=True,
        )
        return
    origin_labels = {"PDF": "📄 PDF", "PubMed": "🌍 PubMed", "Tạp chí VN": "🇻🇳 Tạp chí VN", "Khác": "❓ Khác"}
    pieces = []
    for origin, count in summary["by_origin_sources"].items():
        label = origin_labels.get(origin, origin)
        n_chunks = summary["by_origin_chunks"].get(origin, 0)
        pieces.append(f"{label}: <b>{count}</b> nguồn ({n_chunks} đoạn)")
    st.markdown(
        f'<div class="success-box">✅ <b>Evidence Database{" - " + context_label if context_label else ""}:</b> '
        f'{summary["total_sources"]} nguồn / {summary["total_chunks"]} đoạn bằng chứng '
        f'&nbsp;—&nbsp; {" &nbsp;|&nbsp; ".join(pieces)}</div>',
        unsafe_allow_html=True,
    )

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

BASE_SYSTEM_RULES = """
Bạn là trợ lý nghiên cứu khoa học, hỗ trợ viết luận văn Chuyên khoa cấp I ngành Dược lâm sàng.
NGUYÊN TẮC BẮT BUỘC:
1. Tài liệu cung cấp là nguồn bằng chứng ưu tiên duy nhất.
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
NHIỆM VỤ: {task}
CÂU HỎI/TRUY VẤN: {query}
BẰNG CHỨNG: {evidence_text}
"""
    output = call_gemini(prompt)
    if output is None:
        return None, evidence, []
    converted, invalid_tags = replace_source_tags_with_citations(output, evidence)
    st.session_state["last_generated"] = converted
    st.session_state["last_evidence"] = evidence
    return converted, evidence, invalid_tags

def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*%)?", text)

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
    return re.sub(r"[^\w\s%.,-]", "", text).strip()

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

def spelling_and_terminology_check(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"{BASE_SYSTEM_RULES}\nRà soát lỗi chính tả và thuật ngữ y khoa Dược lâm sàng trong đoạn văn sau:\n{text}"
    return call_gemini(prompt, model=MODEL_LITE)

def plagiarism_style_review(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"{BASE_SYSTEM_RULES}\nĐóng vai hội đồng phản biện luận văn, đánh giá cấu trúc, văn phong học thuật và đối chiếu đạo văn nội bộ với đoạn văn sau:\n{text}"
    return call_gemini(prompt)

def heuristic_ai_style_score(text: str) -> Optional[str]:
    if not text.strip(): return None
    prompt = f"{BASE_SYSTEM_RULES}\nPhân tích các dấu hiệu văn bản có khả năng do AI viết trong đoạn sau:\n{text}"
    return call_gemini(prompt, model=MODEL_LITE)

# ============================================================
# 19. GIAO DIỆN CHÍNH
# ============================================================

st.title("🔬 HỖ TRỢ NGHIÊN CỨU KHOA HỌC")
st.caption("Evidence-Based RAG • Citation Registry • Statistical Engine • Academic Platform")

tabs = st.tabs([
    "📚 1. Tài liệu (PDF)",
    "🔍 2. Tra cứu đa nguồn",
    "✍️ 3. Viết luận văn",
    "📊 4. Phân tích số liệu & Cấu trúc Ch.3",
    "🔎 5. Audit",
    "⚙️ 6. Nguồn & cấu hình",
])

with tabs[0]:
    st.header("📚 Ngân hàng tài liệu gốc (PDF)")
    render_evidence_database_status()
    uploaded_files = st.file_uploader("Tải PDF nghiên cứu / guideline", type=["pdf"], accept_multiple_files=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📥 Nạp tài liệu vào Evidence Database", type="primary"):
            if not uploaded_files:
                st.warning("Chưa có file PDF.")
            else:
                with st.spinner("Đang xử lý PDF và Embedding..."):
                    ns, nc, errors = add_pdf_documents(uploaded_files)
                st.success(f"Đã thêm {ns} tài liệu, {nc} phân đoạn bằng chứng.")
    with col2:
        if st.button("🗑️ Xóa toàn bộ ngân hàng tài liệu"):
            st.session_state["documents"] = {}
            st.session_state["chunks"] = []
            st.session_state["embeddings"] = None
            st.session_state["citation_registry"] = {}
            st.success("Đã xóa dữ liệu.")
            st.rerun()

    docs = list(st.session_state["documents"].values())
    if docs:
        st.dataframe(pd.DataFrame(docs))

with tabs[1]:
    st.header("🔍 Tra cứu đa nguồn (PubMed & Tạp chí VN)")
    render_evidence_database_status()
    t3_query = st.text_input("Tên đề tài nghiên cứu:", placeholder="VD: Khảo sát sử dụng kháng sinh...")
    if st.button("🚀 Tra cứu", type="primary"):
        if t3_query.strip():
            en_query = translate_query_to_mesh(t3_query)
            st.session_state["t3_pm_data"] = search_pubmed(en_query, 5)
            vn_results, _ = search_vn_journals(t3_query, 5)
            st.session_state["t3_vn_data"] = vn_results
    if st.session_state["t3_pm_data"] or st.session_state["t3_vn_data"]:
        if st.button("➕ Nạp tất cả vào Evidence Database"):
            for art in st.session_state["t3_pm_data"]: ingest_pubmed_article(art)
            for art in st.session_state["t3_vn_data"]: ingest_vn_article(art)
            rebuild_index()
            st.success("Đã nạp thành công.")

with tabs[2]:
    st.header("✍️ Viết luận văn dựa trên bằng chứng")
    render_evidence_database_status()
    my_research_data = st.text_area("Bridge Data (Số liệu thực tế của anh từ Tab 4):", height=120)
    citation_rules = "Quy tắc: Dùng SOURCE_TAG, trích dẫn chuẩn, không tự bịa số."
    
    ket_qua_container = st.container()
    def run_quick_task(task_label, query, task_prompt, k):
        with st.spinner(f"AI đang soạn {task_label}..."):
            output, evidence, invalid = generate_evidence_based(task_prompt, query, k=k)
        if output:
            with ket_qua_container:
                st.write("---")
                st.subheader(task_label)
                st.markdown(output)
                bib = citation_bibliography()
                with st.expander("📖 Danh mục tham khảo phiên"):
                    st.code(bib, language="text")

    c1, c2, c3, c4 = st.columns(4)
    with c1: btn_dv = st.button("Đặt vấn đề", use_container_width=True)
    with c2: btn_tq = st.button("Tổng quan", use_container_width=True)
    with c3: btn_pp = st.button("Phương pháp", use_container_width=True)
    with c4: btn_bl = st.button("Bàn luận", use_container_width=True)

    if btn_dv: run_quick_task("Đặt vấn đề", "Tính cấp thiết nghiên cứu", f"Viết Đặt vấn đề.\n{citation_rules}", 6)
    if btn_tq: run_quick_task("Tổng quan", "Tổng quan y văn liên quan", f"Viết Tổng quan.\n{citation_rules}", 8)
    if btn_pp: run_quick_task("Phương pháp", "Phương pháp nghiên cứu", f"Viết Phương pháp.\n{citation_rules}", 5)
    if btn_bl and my_research_data.strip(): run_quick_task("Bàn luận", my_research_data, f"Viết Bàn luận dựa trên số liệu thực tế.\n{citation_rules}", 8)

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

with tabs[4]:
    st.header("🔎 Audit luận văn")
    audit_text = st.text_area("Dán đoạn văn cần kiểm tra", height=250)
    c_a1, c_a2, c_a3 = st.columns(3)
    with c_a1:
        if st.button("🔢 Audit số liệu"):
            if audit_text: st.write(audit_generated_text(audit_text))
    with c_a2:
        if st.button("📄 Kiểm tra nguy cơ đạo văn"):
            if audit_text:
                overlaps = internal_overlap_audit(audit_text, top_k=5)
                st.write(f"Tỷ lệ trùng n-gram cao nhất phát hiện trong kho: {max([o['similarity'] for o in overlaps])*100 if overlaps else 0}%")
                st.markdown(plagiarism_style_review(audit_text) or "")
    with c_a3:
        if st.button("🤖 Chỉ báo AI-style"):
            if audit_text: st.markdown(heuristic_ai_style_score(audit_text) or "")

with tabs[5]:
    st.header("⚙️ Nguồn, Citation & Quản lý Dự án")
    st.write(f"**Gemini Model:** `{DEFAULT_MODEL}`")
    st.write(f"**Citation Registry hiện có:** {len(st.session_state['citation_registry'])} nguồn")
    st.code(citation_bibliography(), language="text")

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
    if st.button("📄 Xuất bản nháp ra Word"):
        docx_data = create_word_document("Bản nháp luận văn", st.session_state["last_generated"], citation_bibliography())
        st.download_button("📥 Tải Word", data=docx_data, file_name="Ban_nhap_CKI.docx")
