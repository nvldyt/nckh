# app.py
# ============================================================
# HỖ TRỢ NGHIÊN CỨU KHOA HỌC – EVIDENCE-BASED RAG
# Bản tối ưu cho luận văn Chuyên khoa cấp I – Dược lâm sàng
# Kiến trúc Modular 4 Engines (Tích hợp Reranker & Study Context)
# ============================================================

import io
import os
import re
import time
import json
import gc
from typing import Any, Dict, List, Optional, Tuple
from chat_data_engine import render_chat_assistant
from chat_writing_engine import render_writing_chat

import numpy as np
import pandas as pd
import streamlit as st
from docx import Document

# ============================================================
# IMPORT TỪ CÁC MODULE CHUYÊN MÔN
# ============================================================
from table_selection_engine import StudyObjective, CandidateResult, TableSelectionEngine, NarrativePlanner, Priority, Presentation
from statistical_engine import validate_dataframe, descriptive_table, numeric_summary, crosstab_test, compare_two_groups, binary_logistic_regression, create_clinical_groups, generate_baseline_table
from evidence_engine import SourceDocument, EvidenceChunk, get_serpapi_key, extract_pdf, search_pubmed, search_vn_journals, ingest_pubmed_article, ingest_vn_article, add_source_and_chunks
from project_storage import save_project, load_project, list_projects, delete_project
from data_engine import auto_clean_data

# ============================================================
# IMPORT 4 ENGINE CỐT LÕI AI 
# ============================================================
from citation_engine import CitationEngine
from audit_engine import Audit_generated_text, internal_overlap_Audit
from retrieval_engine import build_bm25_index, build_embedding_index, update_embedding_index, retrieve_evidence
from writing_engine import call_gemini, generate_evidence_based, BASE_SYSTEM_RULES, MODEL_LITE, DEFAULT_MODEL
from synthesis_engine import build_literature_matrix
from chapter_assembler_engine import assemble_results_and_discussion_chapter

# ============================================================
# 1. CẤU HÌNH GIAO DIỆN & STATE
# ============================================================
st.set_page_config(page_title="NCKH", page_icon="🔬", layout="wide")

DEFAULT_TOP_K = 8
MAX_TOP_K = 20
DEFAULT_VN_JOURNAL_DOMAINS = [
    "tapchiyhocvietnam.vn", "vjol.info", "tapchinghiencuuyhoc.vn", 
    "jmp.huemed-univ.edu.vn", "jmpm.vn", "huejmp.vn", 
    "tcydls108.benhvien108.vn", "tapchiyhcd.vn", "thaibinhjmp.vn", "hup.edu.vn",
]

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li, .stMarkdown { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8f9fa; color: #2c3e50; }
    h1 { color: #1e293b !important; text-align: center; font-weight: 800; margin-bottom: 6px; }
    .stTabs [data-baseweb="tab-panel"] { background: #ffffff; border-radius: 12px; padding: 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-top: 10px; }
    .stTabs [data-baseweb="tab-list"] { background: #f1f5f9; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px !important; font-weight: 600; color: #64748b; }
    .stTabs [aria-selected="true"] { background: #ffffff !important; color: #2563eb !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    div.stButton > button { background-color: #ffffff !important; color: #334155 !important; font-weight: 600; border-radius: 8px; border: 1px solid #cbd5e1 !important; }
    div.stButton > button[kind="primary"] { background-color: #2563eb !important; color: white !important; border: none !important; }
    .warning-box { border-left: 4px solid #f59e0b; padding: 12px 16px; background: #fffbeb; border-radius: 6px; color: #92400e; font-size: 0.95rem;}
    .danger-box  { border-left: 4px solid #ef4444; padding: 12px 16px; background: #fef2f2; border-radius: 6px; color: #991b1b; font-size: 0.95rem;}
    .success-box { border-left: 4px solid #10b981; padding: 12px 16px; background: #ecfdf5; border-radius: 6px; color: #065f46; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

UI_NAMESPACE = "nckh_cki"

def init_state():
    if "ui_version" not in st.session_state: st.session_state["ui_version"] = 0
    defaults = {
        "documents": {}, "chunks": [], "embeddings": None, "bm25": None, "citation_registry": {},
        "Audit_log": [], "last_generated": "", "last_evidence": [], "current_references": [],
        "vn_journal_domains": list(DEFAULT_VN_JOURNAL_DOMAINS), "t3_pm_data": [], "t3_vn_data": [], "t3_query": "",
        "result_cart": [], "saved_tables": {}, "selection_decisions": [], "narrative_plan": {}, "study_context": {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

def ui_key(widget_name: str) -> str:
    return f"{UI_NAMESPACE}_v{st.session_state.get('ui_version', 0)}_{widget_name}"

def reset_ui_state():
    st.session_state["ui_version"] = st.session_state.get("ui_version", 0) + 1


# ============================================================
# 2. CÁC HÀM CẦU NỐI (WRAPPERS) - ĐẨY DỮ LIỆU TỪ UI VÀO ENGINES
# ============================================================

# --- WRAPPER CHO RAG ---
def rebuild_index(new_chunks: List[Dict[str, Any]] = None):
    all_chunks = st.session_state.get("chunks", [])
    if not all_chunks:
        st.session_state["embeddings"] = None
        st.session_state["bm25"] = None
        return
    if new_chunks and st.session_state.get("embeddings") is not None:
        st.session_state["embeddings"] = update_embedding_index(new_chunks, st.session_state["embeddings"])
    else:
        st.session_state["embeddings"] = build_embedding_index(all_chunks)
    st.session_state["bm25"] = build_bm25_index(all_chunks)

def retrieve_evidence_wrapper(query: str, k: int = 8) -> List[Dict[str, Any]]:
    return retrieve_evidence(
        query=query,
        chunks=st.session_state.get("chunks", []),
        matrix=st.session_state.get("embeddings"),
        bm25=st.session_state.get("bm25"),
        top_k=k
    )

# --- WRAPPER CHO CITATION ---
def get_citation_engine() -> CitationEngine:
    if "citation_engine" not in st.session_state:
        st.session_state["citation_engine"] = CitationEngine()
    return st.session_state["citation_engine"]

def source_metadata(source_id: str) -> Dict[str, Any]:
    return st.session_state["documents"].get(source_id, {})

def citation_bibliography_wrapper() -> str:
    refs = st.session_state.get("current_references", [])
    docs = st.session_state.get("documents", {})
    rows = []
    for ref in refs:
        source_id = ref['ref_id'].replace("REF-", "") if ref['ref_id'].startswith("REF-") else ref['ref_id']
        meta = docs.get(source_id, ref.get("metadata", {}))
        authors = meta.get('authors') or "Tác giả chưa xác định"
        title = meta.get('title') or meta.get('file_name') or "Tài liệu chưa xác định"
        journal = meta.get('journal') or "Tài liệu lưu trữ"
        year = meta.get('year') or "Năm chưa rõ"
        citation = f"[{ref['vancouver_index']}] {authors}. {title}. {journal}. {year}."
        if meta.get("doi"): citation += f" DOI: {meta['doi']}."
        if meta.get("pmid"): citation += f" PMID: {meta['pmid']}."
        if meta.get("url") and meta.get("origin") == "Tạp chí VN": citation += f" [{meta['url']}]"
        rows.append(citation)
    return "\n".join(rows)

# --- WRAPPER CHO WRITING & METADATA ---
def generate_evidence_based_wrapper(task: str, query: str, k: int = 8) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    evidence = retrieve_evidence_wrapper(query, k=k)
    engine = get_citation_engine()
    study_ctx = st.session_state.get("study_context", {})
    
    final_text, references, invalid_tags = generate_evidence_based(
        task_prompt=task, 
        evidence=evidence, 
        citation_engine=engine,
        study_context=study_ctx
    )
    
    if final_text:
        st.session_state["last_generated"] = final_text
        st.session_state["last_evidence"] = evidence
        st.session_state["current_references"] = references
    return final_text, evidence, invalid_tags

def extract_metadata_from_text_ai_wrapper(text: str) -> dict:
    prompt = f"""Bạn là chuyên gia thư viện y khoa. Nhiệm vụ của bạn là trích xuất siêu dữ liệu (metadata) từ văn bản thô của trang đầu tiên của một bài báo nghiên cứu.
    
ĐẶC BIỆT LƯU Ý VỚI BÀI BÁO TIẾNG VIỆT:
1. Tác giả (authors): Thường nằm ngay dưới tiêu đề bài báo. Hãy lọc bỏ tên cơ quan/bệnh viện/đại học. Gom các tên người lại, cách nhau bằng dấu phẩy (VD: Nguyễn Văn A, Trần Thị B).
2. Tạp chí (journal): Tìm các cụm từ bắt đầu bằng "Tạp chí", "Y học", "Nghiên cứu", "Y dược", "Journal".
3. Năm xuất bản (year): Tìm con số 4 chữ số hợp lý nhất (VD: 2021, 2023).

TRẢ VỀ DUY NHẤT MỘT CHUỖI JSON HỢP LỆ, KHÔNG GIẢI THÍCH GÌ THÊM.
Cấu trúc JSON bắt buộc:
{{
    "authors": "...",
    "title": "...",
    "year": "...",
    "journal": "...",
    "doi": "..."
}}
Nếu không tìm thấy trường nào, hãy để chuỗi rỗng "". 

ĐOẠN VĂN BẢN QUÉT ĐƯỢC TỪ TRANG 1:
{text[:4500]}"""

    # Gọi mô hình mạnh thay vì mô hình Lite, ép nhiệt độ = 0 để tăng tính chính xác tuyệt đối
    res = call_gemini(prompt, model=DEFAULT_MODEL, temperature=0.0)
    if not res: return {}
    try:
        cleaned = res.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        elif cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except: return {}

# --- WRAPPER CHO AUDIT ---
def Audit_generated_text_wrapper(text: str) -> Dict[str, Any]:
    relevant_evidence = retrieve_evidence_wrapper(text, k=6)
    return Audit_generated_text(text, relevant_evidence)

def internal_overlap_Audit_wrapper(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return internal_overlap_Audit(text, st.session_state.get("chunks", []), top_k=top_k)

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
import gc # Đảm bảo anh đã khai báo import gc ở đầu file app.py nhé

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
                
                # Lấy source_id an toàn (hỗ trợ cả dict và object)
                source_id = source.get("source_id") if isinstance(source, dict) else getattr(source, "source_id", None)
                
                if source_id and chunks:
                    # Lọc nội dung trang 1 an toàn cho EvidenceChunk object
                    page_one_chunks = []
                    for c in chunks:
                        c_text = c.get("text") if isinstance(c, dict) else getattr(c, 'text', '')
                        c_page = str(c.get("page") if isinstance(c, dict) else getattr(c, 'page', '')).lower()
                        if c_page in ["1", "trang 1", "page 1"]:
                            page_one_chunks.append(c_text)
                    
                    # Nếu không tìm thấy nhãn trang 1, lấy tạm chunk đầu tiên
                    if not page_one_chunks:
                        first_chunk = chunks[0]
                        target_text = first_chunk.get("text") if isinstance(first_chunk, dict) else getattr(first_chunk, 'text', '')
                    else:
                        target_text = page_one_chunks[0]
                    
                    if target_text:
                        # Gọi AI trích xuất thông tin tác giả, tiêu đề, năm... ngay lúc nạp
                        meta_ai = extract_metadata_from_text_ai_wrapper(target_text)
                        if meta_ai:
                            current_docs = st.session_state.get("documents", {})
                            if source_id in current_docs:
                                current_docs[source_id].update({k: v for k, v in meta_ai.items() if v})
                                
            # ==========================================
            # DỌN RÁC GIẢI PHÓNG RAM CHO TỪNG FILE
            # ==========================================
            del source
            gc.collect() 
            
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")
            
    if new_sources: 
        rebuild_index(new_chunks=new_chunks_list)
        
    return new_sources, new_chunks_count, errors

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
        "total_sources": len(documents), "total_chunks": len(chunks),
        "by_origin_sources": by_origin_sources, "by_origin_chunks": by_origin_chunks,
        "index_ready": st.session_state.get("embeddings") is not None,
    }

def render_evidence_database_status(context_label: str = ""):
    summary = evidence_database_summary()
    if summary["total_sources"] == 0:
        st.markdown('<div class="danger-box">⚠️ <b>Evidence Database đang RỖNG.</b> Hãy nạp PDF ở Tab 1.</div>', unsafe_allow_html=True)
        return
    origin_labels = { "PDF": "📄 PDF", "PubMed": "🌍 PubMed", "Tạp chí VN": "🇻🇳 Tạp chí VN", "Khác": "❓ Khác" }
    pieces = []
    for origin, count in summary["by_origin_sources"].items():
        label = origin_labels.get(origin, origin)
        n_chunks = summary["by_origin_chunks"].get(origin, 0)
        pieces.append(f"{label}: <b>{count}</b> nguồn ({n_chunks} đoạn)")
    status_html = (
        f'<div class="success-box">✅ <b>Evidence Database{" - " + context_label if context_label else ""}:</b> '
        f'{summary["total_sources"]} nguồn / {summary["total_chunks"]} đoạn '
        f'&nbsp;—&nbsp; {" &nbsp;|&nbsp; ".join(pieces)}'
        f'{"" if summary["index_ready"] else " &nbsp;—&nbsp; ⚠️ chưa dựng xong index, thử tải lại"}</div>'
    )
    st.markdown(status_html, unsafe_allow_html=True)

def translate_query_to_mesh(vietnamese_query: str) -> str:
    prompt = f"Chuyển đổi tên đề tài tiếng Việt sau thành chuỗi từ khóa tiếng Anh hiệu quả để tra cứu PubMed (Dùng AND, OR, Không dùng dấu gạch chéo). Đề tài: '{vietnamese_query}'"
    text = call_gemini(prompt, model=MODEL_LITE)
    return text.strip().strip('"').strip("'").replace("\n", " ") if text else vietnamese_query

def add_markdown_body_to_doc(doc: Document, body: str):
    lines = body.split("\n")
    buffer_paragraph = []
    def flush_paragraph():
        if buffer_paragraph:
            doc.add_paragraph(" ".join(buffer_paragraph).strip())
            buffer_paragraph.clear()
    for line in lines:
        stripped = line.strip()
        is_table_row = stripped.startswith("|")
        if not is_table_row: doc._last_table_open = False
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("### "): flush_paragraph(); doc.add_heading(stripped[4:].strip(), level=3)
        elif stripped.startswith("## "): flush_paragraph(); doc.add_heading(stripped[3:].strip(), level=2)
        elif stripped.startswith("# "): flush_paragraph(); doc.add_heading(stripped[2:].strip(), level=1)
        elif stripped.startswith(("- ", "* ")): flush_paragraph(); doc.add_paragraph(stripped[2:].strip(), style="List Bullet")
        elif is_table_row:
            if set(stripped.replace("|", "").replace(" ", "").replace(":", "")) <= {"-"}: continue
            flush_paragraph()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            table = doc.tables[-1] if (doc.tables and getattr(doc, "_last_table_open", False)) else None
            if table is None:
                table = doc.add_table(rows=1, cols=len(cells))
                table.style = "Light Grid Accent 1"
                for i, c in enumerate(cells): table.rows[0].cells[i].text = c
                doc._last_table_open = True
            else:
                row = table.add_row()
                for i, c in enumerate(cells):
                    if i < len(row.cells): row.cells[i].text = c
        else: buffer_paragraph.append(stripped)
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
# 4. GIAO DIỆN CHÍNH STREAMLIT
# ============================================================
def main():
    init_state()
    st.title("🔬 HỖ TRỢ NGHIÊN CỨU KHOA HỌC")
    st.caption("Evidence-Based RAG • Tra cứu TLTK • Citation Registry • Statistical Engine • Audit")
    # ============================================================
    # SIDEBAR - QUẢN LÝ DỰ ÁN (Chuyển từ Tab 6 cũ ra ngoài)
    # ============================================================
    with st.sidebar:
        st.header("⚙️ Quản lý dự án")
        
        if st.button("🧹 Làm mới Giao diện", use_container_width=True): 
            reset_ui_state()
            st.rerun()
            
        st.divider()
        st.subheader("💾 Lưu & Khôi phục (.json)")
        
        # Nút tải xuống
        project_data = {
            "documents": st.session_state.get("documents", {}), 
            "chunks": st.session_state.get("chunks", []),
            "citation_registry": st.session_state.get("citation_registry", {}), 
            "current_references": st.session_state.get("current_references", []),
            "result_cart": [vars(item) if hasattr(item, "__dict__") else item for item in st.session_state.get("result_cart", [])],
            "saved_tables": {k: v.to_dict(orient='split') for k, v in st.session_state.get("saved_tables", {}).items()}
        }
        st.download_button("📥 Tải file dự án", data=json.dumps(project_data, ensure_ascii=False, indent=4), file_name="Du_An_Luan_Van.json", mime="application/json", use_container_width=True)
        
        # Nút tải lên
        uploaded_proj = st.file_uploader("Khôi phục từ file:", type=["json"])
        if uploaded_proj and st.button("🚀 Khôi phục", type="primary", use_container_width=True):
            try:
                loaded_data = json.load(uploaded_proj)
                fallback_defaults = {"documents": {}, "chunks": [], "citation_registry": {}, "current_references": []}
                st.session_state.update({k: loaded_data.get(k, fallback_defaults[k]) for k in fallback_defaults.keys()})
                
                # Phục hồi object phức tạp an toàn
                if "CandidateResult" in globals():
                    st.session_state["result_cart"] = [CandidateResult(**item) if isinstance(item, dict) else item for item in loaded_data.get("result_cart", [])]
                st.session_state["saved_tables"] = {k: pd.DataFrame.from_dict(v, orient='split') for k, v in loaded_data.get("saved_tables", {}).items()}
                
                with st.spinner("Đang xây dựng lại index..."): 
                    rebuild_index()
                st.success("🎉 Khôi phục thành công!")
                time.sleep(1)
                reset_ui_state()
                st.rerun()
            except Exception as e: 
                st.error(f"❌ Lỗi: {str(e)}")
    tabs = st.tabs([
        "📚 1. Tài liệu (PDF)", 
        "🔍 2. Tra cứu TLTK", 
        "✍️ 3. Viết luận văn", 
        "📊 4. Phân tích số liệu", 
        "🔎 5. Kiểm tra luận văn", 
        "🏷️ 6. Trích dẫn TLTK" # <-- Thêm Tab mới này
    ])
    # ------------------------------------------------------------
    # TAB 1 – TÀI LIỆU PDF
    # ------------------------------------------------------------
    with tabs[0]:
        st.header("📚 Ngân hàng tài liệu gốc (PDF)")
        render_evidence_database_status()

        uploaded_files = st.file_uploader("Tải PDF nghiên cứu / guideline / bài báo", type=["pdf"], accept_multiple_files=True, key=ui_key("pdf_uploader"))
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📥 Nạp tài liệu vào Evidence Database", type="primary"):
                if not uploaded_files: st.warning("Chưa có file PDF.")
                else:
                    with st.spinner("Đang đọc PDF, tạo source registry và embedding..."):
                        ns, nc, errors = add_pdf_documents(uploaded_files)
                    st.success(f"Đã thêm {ns} tài liệu, {nc} phân đoạn bằng chứng.")
                    for err in errors: st.error(err)
        with col2:
            if st.button("🗑️ Xóa toàn bộ ngân hàng tài liệu"):
                st.session_state["documents"] = {}; st.session_state["chunks"] = []; st.session_state["embeddings"] = None
                st.session_state["bm25"] = None; st.session_state["citation_registry"] = {}; st.session_state["last_evidence"] = []
                st.success("Đã xóa dữ liệu trong phiên hiện tại."); st.rerun()

        st.write("---")
        st.subheader("Nguồn PDF đã nạp")
        docs = list(st.session_state["documents"].values())
        if docs: st.dataframe(pd.DataFrame(docs))
        else: st.info("Chưa có tài liệu.")

        st.subheader("Tìm bằng chứng trong toàn bộ Evidence Database")
        evidence_query = st.text_area("Nhập vấn đề cần tìm trong tài liệu:", placeholder="Ví dụ: tỷ lệ bệnh nhân đạt huyết áp mục tiêu...", key=ui_key("evidence_query"))
        top_k = st.slider("Số đoạn bằng chứng", 3, MAX_TOP_K, DEFAULT_TOP_K, key=ui_key("top_k_tab1"))

        if st.button("🔎 Truy xuất bằng chứng", key="retrieve_tab1"):
            if not evidence_query.strip(): st.warning("Nhập câu hỏi trước.")
            else:
                evidence = retrieve_evidence_wrapper(evidence_query, k=top_k)
                st.session_state["last_evidence"] = evidence
                if not evidence: st.warning("Không tìm thấy bằng chứng trong tài liệu.")
                else:
                    for ev in evidence:
                        meta = st.session_state["documents"].get(ev["source_id"], {})
                        st.markdown(f"**{ev['chunk_id']}** _({meta.get('origin', '')})_\nNguồn: {ev.get('file_name','')} — Trang/mục: {ev.get('page','')}\nĐiểm: {ev.get('score', 0):.4f}\n\n> {ev['text']}")
        st.write("---")

    # ------------------------------------------------------------
    # TAB 2 – Tra cứu TLTK
    # ------------------------------------------------------------
    with tabs[1]:
        st.header("🔍 Tra cứu TLTK: PubMed + Bài báo khoa học")
        st.info("Nhập tên đề tài bằng tiếng Việt. Hệ thống tự dịch sang từ khoá MeSH để tìm trên PubMed, đồng thời tìm bài báo tiếng Việt liên quan.")
        render_evidence_database_status()

        col_search, col_btn = st.columns([4, 1])
        with col_search: t3_query = st.text_input("Tên đề tài nghiên cứu (tiếng Việt):", key=ui_key("t3_query_input"))
        with col_btn: max_res = st.number_input("Số bài/nguồn", min_value=2, max_value=10, value=5, key=ui_key("t3_max_res"))
        
        # Đã thụt lề toàn bộ khối if này vào trong (thêm 4 dấu cách)
        if st.button("🚀 Tra cứu TLTK", type="primary", key="t3_btn_search"):
            if not t3_query.strip():
                st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
            else:
                st.session_state["t3_query"] = t3_query
                
                with st.spinner("Đang dịch & chuẩn hoá từ khoá sang MeSH (PubMed)..."):
                    en_query = translate_query_to_mesh(t3_query)
                    st.session_state["t3_en_keyword"] = en_query
                    
                with st.spinner("Đang tìm & tải Abstract từ PubMed..."):
                    # Cắt bỏ phần rườm rà bằng cả '###' hoặc '---'
                    clean_en_query = en_query.split("###")[0].split("---")[0].strip()
                    
                    # Nếu chuỗi sau khi cắt vẫn chứa tiếng Việt hoặc quá lộn xộn, dùng luôn từ khóa gốc của anh
                    vietnamese_markers = ["hoặc", "nếu", "về", "trong", "đề tài", "từ khóa"]
                    if not clean_en_query or any(marker in clean_en_query.lower() for marker in vietnamese_markers):
                        clean_en_query = t3_query.strip()
                        
                    st.session_state["t3_pm_data"] = search_pubmed(clean_en_query, max_res)
                    
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
                if st.session_state.get("t3_vn_keyword"): st.success(f"🔑 Từ khoá đã rút gọn: **{st.session_state['t3_vn_keyword']}**")
                if not st.session_state.get("t3_vn_data"): st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
                else:
                    for i, art in enumerate(st.session_state["t3_vn_data"]):
                        with st.container(border=True):
                            st.markdown(f"**[{art['title']}]({art['link']})**" if art.get("link") else f"**{art['title']}**")
                            st.caption(art["source"])
                            st.write(art["snippet"])
                        if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"vn_ingest_{i}")):
                            if ingest_vn_article(art): 
                                rebuild_index()
                                st.success("Đã nạp. Nhớ Audit bản gốc trước khi dùng số liệu chi tiết.")
                                st.rerun() # <--- Thêm dòng này để cập nhật giao diện ngay lập tức
                            else: 
                                st.info("Nguồn này đã có trong Evidence Database.")
            with col_pm:
                st.markdown("### 🌍 PubMed (Quốc tế)")
                if st.session_state.get("t3_en_keyword"): st.success(f"🔑 Từ khoá MeSH: **{st.session_state['t3_en_keyword']}**")
                if not st.session_state.get("t3_pm_data"): st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
                else:
                    for i, art in enumerate(st.session_state["t3_pm_data"]):
                        with st.container(border=True):
                            st.markdown(f"**[{art['title']}]({art['url']})**")
                            st.caption(f"✍️ {art['authors']} ({art['year']}) — {art['journal']}")
                            with st.expander("Xem tóm tắt (Abstract)"): st.write(art["abstract"])
                        if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"pm_ingest_{i}")):
                            if ingest_pubmed_article(art): 
                                rebuild_index()
                                st.success("Đã nạp vào Evidence Database.")
                                st.rerun() # <--- Thêm dòng này để cập nhật giao diện ngay lập tức
                            else: 
                                st.info("Nguồn này đã có trong Evidence Database.")

            st.write("---")
            if st.button("➕ Nạp TẤT CẢ kết quả ở trên vào Evidence Database", key="t3_ingest_all"):
                count = sum(1 for art in st.session_state.get("t3_pm_data", []) if ingest_pubmed_article(art))
                count += sum(1 for art in st.session_state.get("t3_vn_data", []) if ingest_vn_article(art))
                if count: rebuild_index()
                st.success(f"Đã nạp {count} nguồn mới vào Evidence Database.")

    # ------------------------------------------------------------
    # TAB 3 – VIẾT LUẬN VĂN
    # ------------------------------------------------------------
    with tabs[2]:
        st.header("📝 Viết tự động bằng AI (RAG)")
        st.header("✍️ Viết luận văn dựa trên bằng chứng")
        st.warning("Đây là công cụ tạo bản nháp. Mọi citation và số liệu phải được Audit lại (đối chiếu bản gốc) trước khi đưa vào luận văn chính thức.")
        render_evidence_database_status("dùng cho các nút viết nhanh bên dưới")

        # ============================================================
        # KHAI BÁO BỐI CẢNH NGHIÊN CỨU (STUDY CONTEXT)
        # ============================================================
        with st.expander("🎯 KHAI BÁO BỐI CẢNH NGHIÊN CỨU (STUDY CONTEXT) - Cấu hình 1 lần dùng mãi mãi", expanded=True):
            st.info("💡 Khai báo thông tin đề tài tại đây để AI tự động hiểu và bám sát vào mục tiêu của anh trong mọi lần sinh văn bản, không sợ lạc đề.")
            ctx = st.session_state.get("study_context", {})
            c1, c2 = st.columns(2)
            with c1:
                ctx_title = st.text_input("Tên đề tài:", value=ctx.get("title", ""), placeholder="VD: Phân tích tình hình sử dụng thuốc...")
                ctx_design = st.text_input("Thiết kế nghiên cứu:", value=ctx.get("design", ""), placeholder="VD: Mô tả cắt ngang hồi cứu")
                ctx_population = st.text_input("Đối tượng bệnh nhân:", value=ctx.get("population", ""), placeholder="VD: Bệnh nhân tăng huyết áp ngoại trú")
            with c2:
                ctx_sample = st.text_input("Cỡ mẫu dự kiến (N=):", value=ctx.get("sample_size", ""), placeholder="VD: 150 bệnh án")
                ctx_obj = st.text_area("Mục tiêu chính:", value=ctx.get("objectives", ""), height=110, placeholder="VD: 1. Khảo sát đặc điểm... 2. Đánh giá tính hợp lý...")
                
            if st.button("💾 Lưu Study Context"):
                st.session_state["study_context"] = {
                    "title": ctx_title, "design": ctx_design, "population": ctx_population,
                    "sample_size": ctx_sample, "objectives": ctx_obj
                }
                st.success("✅ Đã lưu bối cảnh! Bộ não AI đã được đồng bộ hóa với đề tài của anh.")

        # ============================================================
        # TÍNH NĂNG MỚI: MA TRẬN TỔNG HỢP Y VĂN (LITERATURE SYNTHESIS MATRIX)
        # ============================================================
        with st.expander("🌟 Tự động lập Ma trận tổng hợp y văn từ Evidence Database", expanded=False):
            st.info("💡 Tính năng này tự động quét tất cả các tài liệu / bài báo bạn đã nạp, tổng hợp thành bảng so sánh chuẩn y khoa (Tác giả, Năm, Thiết kế, Cỡ mẫu, Kết quả chính).")
            
            if st.button("🚀 Khởi tạo Ma trận Tổng hợp Y văn", type="primary", key="btn_build_matrix"):
                docs = st.session_state.get("documents", {})
                chunks = st.session_state.get("chunks", [])
                
                if not docs:
                    st.warning("⚠️ Evidence Database đang trống! Hãy nạp tài liệu PDF hoặc bài báo từ PubMed/Tạp chí VN trước.")
                else:
                    with st.spinner("AI đang phân tích và cấu trúc hóa ma trận y văn..."):
                        matrix_df = build_literature_matrix(docs, chunks)
                        
                    if not matrix_df.empty:
                        st.success("✅ Đã lập thành công Ma trận tổng hợp y văn!")
                        st.dataframe(matrix_df, use_container_width=True)
                        
                        st.session_state["literature_matrix_df"] = matrix_df
                        
                        md_table = "### Ma trận tổng hợp y văn\n\n" + matrix_df.to_markdown(index=False)
                        matrix_word_bytes = create_word_document(
                            title="Ma trận Tổng hợp Y văn",
                            body=md_table,
                            bibliography=""
                        )
                        st.download_button(
                            label="📥 Tải Ma trận Y văn ra file Word",
                            data=matrix_word_bytes,
                            file_name="Ma_tran_Tong_hop_Y_van.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_matrix_word"
                        )
                    else:
                        st.error("❌ Không thể trích xuất dữ liệu ma trận. Vui lòng thử lại.")
        
        st.markdown("### 📊 Dữ liệu nghiên cứu của riêng anh")
        if "ai_pending_remark" in st.session_state:
            st.session_state[ui_key("my_table_remarks")] = st.session_state["ai_pending_remark"]
            del st.session_state["ai_pending_remark"]

        col_data_1, col_data_2 = st.columns(2)
        with col_data_1:
            my_research_data = st.text_area("1. Số liệu bảng (Dán bảng Excel/Markdown vào đây):", height=180, key=ui_key("my_research_data"))
        with col_data_2:
            my_table_remarks = st.text_area("2. Nhận xét bảng (Chỉ diễn giải số liệu, KHÔNG bàn luận):", height=180, key=ui_key("my_table_remarks"))

        if st.button("✍️ AI Viết Nhận Xét Bảng (Tự động điền vào ô 2)"):
            if not my_research_data.strip(): st.warning("⚠️ Anh cần dán bảng số liệu vào ô số 1 trước để AI có dữ liệu đọc!")
            else:
                with st.spinner("AI đang phân tích bảng và soạn nhận xét..."):
                    task = "Ngắn gọn, CHỈ diễn giải các số liệu nổi bật. KHÔNG bàn luận, KHÔNG so sánh. Viết thành MỘT ĐOẠN VĂN LIỀN MẠCH duy nhất."
                    prompt = f"{BASE_SYSTEM_RULES}\nNHIỆM VỤ:\n{task}\nBẢNG SỐ LIỆU:\n{my_research_data}"
                    generated_remark = call_gemini(prompt)
                    if generated_remark:
                        st.session_state["ai_pending_remark"] = generated_remark
                        st.rerun()

        st.write("---")
        citation_rules = "Chỉ dùng SOURCE_TAG thật để hệ thống tự chuyển thành [n]. Các số trích dẫn theo thứ tự xuất hiện."
        ket_qua_container = st.container()

        def run_quick_task(task_label: str, query: str, task_prompt: str, k: int):
            with st.spinner(f"AI đang soạn: {task_label}..."):
                output, evidence, invalid = generate_evidence_based_wrapper(task_prompt, query, k=k)
                if output:
                    with ket_qua_container:
                        st.write("---")
                        st.subheader(task_label)
                        st.markdown(output)
                        st.markdown("### 🔎 Dấu vết bằng chứng (Evidence Trace)")
                        current_refs = st.session_state.get("current_references", [])
                        if current_refs:
                            for ref in current_refs:
                                v_index = ref['vancouver_index']
                                ref_id = ref['ref_id']
                                source_id = ref_id.replace("REF-", "") if ref_id.startswith("REF-") else ref_id
                                related_chunks = [ev for ev in evidence if ev['source_id'] == source_id]
                                meta = ref.get('metadata', {})
                                title = meta.get('title', 'Tài liệu chưa có tiêu đề')
                                file_name = meta.get('file_name', 'N/A')
                                with st.expander(f"[{v_index}] ↳ {title[:85]}..."):
                                    st.write(f"**Tệp gốc:** `{file_name}`")
                                    for chunk in related_chunks:
                                        st.markdown(f"- **Trang/Mục:** `{chunk.get('page', 'N/A')}` | **Độ khớp:** `{chunk.get('score', 0):.4f}`")
                                        st.info(f"_{chunk.get('text', '')}_")
                        else: st.info("Đoạn văn này không sử dụng trích dẫn nào từ Evidence Database.")

                        bib = citation_bibliography_wrapper()
                        with st.expander("📖 Danh mục Tài liệu tham khảo (Của bản nháp này)"): st.code(bib if bib else "Chưa có citation registry.", language="text")

                        Audit = Audit_generated_text_wrapper(output)
                        colA, colB = st.columns(2)
                        with colA:
                            if invalid: st.error(f"Phát hiện citation ảo: {', '.join(invalid)}")
                            else: st.success("Không phát hiện citation ảo.")
                        with colB:
                            if Audit.get("warnings"): st.warning(f"Số liệu lạ (Cần Audit lại): {', '.join(Audit['warnings'])}")
                            else: st.success("Không phát hiện số liệu lạ ngoài bằng chứng.")
                        st.session_state["Audit_log"].append({"type": task_label, "invalid_citation": invalid, "Audit": Audit})

        st.subheader("📝 Lệnh viết nhanh")
        c1, c2, c3, c4 = st.columns(4)
        with c1: btn_dat_van_de = st.button("Đặt vấn đề")
        with c2: btn_tong_quan = st.button("Tổng quan tài liệu")
        with c3: btn_phuong_phap = st.button("Phương pháp NC")
        with c4: btn_ban_luan = st.button("Bàn luận KQNC và So sánh")
      
        st.write("---")
        st.subheader("Lệnh tùy chỉnh")
        custom_prompt = st.text_area("Nhập câu lệnh khác:", key=ui_key("custom_prompt_tab3"))
        k_custom = st.slider("Số nguồn bằng chứng truy xuất", 3, MAX_TOP_K, DEFAULT_TOP_K, key=ui_key("top_k_tab3"))
        btn_custom = st.button("▶️ Chạy lệnh tùy chỉnh")

        if btn_dat_van_de:
            query = "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, dịch tễ học, gánh nặng bệnh tật liên quan sử dụng thuốc"
            task = f"Viết phần 'Đặt vấn đề'. Viết thành MỘT MẠCH VĂN LIỀN MẠCH, khoảng 400 từ, gồm 3-4 đoạn văn.\n{citation_rules}"
            run_quick_task("Đặt vấn đề", query, task, k=6)
        if btn_tong_quan:
            query = "Tổng quan y văn, các nghiên cứu liên quan, cơ chế dược lý, kết quả chính, khuyến cáo điều trị"
            task = f"Viết phần 'Tổng quan tài liệu' chuyên sâu.\n{citation_rules}"
            run_quick_task("Tổng quan tài liệu", query, task, k=8)
        if btn_phuong_phap:
            query = "Đối tượng nghiên cứu, tiêu chuẩn chọn loại, thiết kế nghiên cứu, cỡ mẫu, biến số nghiên cứu"
            task = f"Viết 'Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU'.\n{citation_rules}"
            run_quick_task("Phương pháp nghiên cứu", query, task, k=5)
        if btn_ban_luan:
            if not my_research_data.strip() and not my_table_remarks.strip():
                st.warning("⚠️ Cần dán bảng số liệu (ô 1) và nhận xét (ô 2) vào phía trên trước!")
            else:
                context = f"SỐ LIỆU BẢNG:\n{my_research_data}\n\nNHẬN XÉT DIỄN GIẢI:\n{my_table_remarks}"
                task = f"DỮ LIỆU NGHIÊN CỨU:\n{context}\nYÊU CẦU: Viết BÀN LUẬN TOÀN DIỆN. Giải thích nguyên nhân và so sánh với y văn. Viết liền mạch, không dùng nhãn phân chia.\n{citation_rules}"
                run_quick_task("Bàn luận và So sánh toàn diện", context, task, k=8)
        
        if btn_custom:
            if not custom_prompt.strip(): st.warning("Vui lòng nhập yêu cầu!")
            else:
                task = f"{custom_prompt}\n{citation_rules}"
                run_quick_task("Kết quả lệnh tùy chỉnh", custom_prompt, task, k=k_custom)
        st.write("---")
        st.subheader("📄 Xuất Bản Nháp")
        if st.button("📥 Tải bản nháp hiện tại ra file Word", use_container_width=True, type="primary"):
            if not st.session_state.get("last_generated"): 
                st.warning("Chưa có bản nháp. Vui lòng chạy một lệnh viết luận văn trước.")
            else:
                docx_data = create_word_document(title="Bản nháp hỗ trợ nghiên cứu", body=st.session_state["last_generated"], bibliography=citation_bibliography_wrapper())
                st.download_button("Bấm vào đây để tải file", data=docx_data, file_name="Ban_nhap.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        render_writing_chat()
    # ------------------------------------------------------------
    # TAB 4 – PHÂN TÍCH SỐ LIỆU & TUYỂN CHỌN BẢNG
    # ------------------------------------------------------------
    with tabs[3]:
        st.header("📊 Phân tích số liệu bệnh án (SPSS Mini)")

        # --- CHÈN HÀM ĐỌC EXCEL AN TOÀN VÀO ĐÂY ---
        def safe_read_excel(uploaded_file):
            import pandas as pd
            import gc
            try:
                df = pd.read_excel(uploaded_file)
                df.columns = df.columns.astype(str) # Ép Header thành chữ
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str) # Khử dữ liệu lẩu thập cẩm
                gc.collect() # Dọn rác RAM
                return df
            except Exception as e:
                st.error(f"❌ Lỗi khi đọc Excel: {e}")
                return pd.DataFrame()

        # --- KHU VỰC TẢI VÀ HIỂN THỊ FILE ---
        # Vẫn giữ nguyên key=ui_key("excel_data") của anh để không lỗi các hàm khác
        excel_file = st.file_uploader("Tải file Excel", type=["xlsx", "xls"], key="excel_uploader_tab4_unique")
        
        if excel_file is not None:
            with st.spinner("Đang dọn dẹp và nạp dữ liệu..."):
                df_clean = safe_read_excel(excel_file)
                
                if not df_clean.empty:
                    st.success(f"✅ Nạp thành công: {df_clean.shape[0]} dòng và {df_clean.shape[1]} cột.")
                    # Chỉ hiển thị 50 dòng đầu tiên để không treo RAM trình duyệt
                    st.dataframe(df_clean.head(50), use_container_width=True)
                    
                    # Cập nhật vào session_state (Anh kiểm tra lại tên biến này cho khớp với code cũ nhé)
                    st.session_state["excel_data"] = df_clean                    

        # Gọi hệ thống Chat Độc lập đã được tách module
        render_chat_assistant()
        
        if excel_file is not None:
            try:
                raw_df = pd.read_excel(excel_file)
                with st.spinner("Đang dọn dẹp và chuẩn hóa dữ liệu bằng Data Engine..."):
                    df, clean_logs = auto_clean_data(raw_df)
                st.session_state["excel_df"] = df
                st.session_state["clean_logs"] = clean_logs
                st.success(f"Dữ liệu sẵn sàng: {df.shape[0]} dòng × {df.shape[1]} cột.")
            except Exception as exc:
                st.error(f"Lỗi khi đọc hoặc xử lý file Excel: {exc}")
                st.session_state["excel_df"] = None

        df = st.session_state.get("excel_df")

        if df is not None and not df.empty:
            clean_logs = st.session_state.get("clean_logs", [])
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
                        rows = ["| " + " | ".join(str(x) for x in row.values) + " |" for _, row in df_table.iterrows()]
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
            # TRÌNH LẮP RÁP TỰ ĐỘNG TOÀN BỘ CHƯƠNG KẾT QUẢ & BÀN LUẬN
            # ==========================================
            st.write("---")
            st.subheader("🚀 Trình lắp ráp tự động Chương Kết quả & Bàn luận (Auto-Assembler Agent)")
            st.info("💡 Hệ thống sẽ tự động kích hoạt Loop Agent: Duyệt qua từng bảng theo mạch kể chuyện, tự viết nhận xét, tự tìm bằng chứng đối chiếu và lắp ráp thành toàn bộ bản thảo 2 chương lớn.")

            if st.button("🪄 Tự động lập bản thảo toàn bộ Chương 3 & Chương 4", type="primary", key="btn_auto_assemble"):
                decisions = st.session_state.get("selection_decisions", [])
                saved_tabs = st.session_state.get("saved_tables", {})
                chunks_db = st.session_state.get("chunks", [])
                embeddings_matrix = st.session_state.get("embeddings")
                bm25_index = st.session_state.get("bm25")
                citation_eng = get_citation_engine()
                study_ctx = st.session_state.get("study_context", {})

                if not decisions or not saved_tabs:
                    st.warning("⚠️ Bạn cần chạy 'Table Selection Engine' để thiết lập mạch kể chuyện và lưu bảng vào Giỏ trước!")
                else:
                    with st.spinner("Agent đang tự động quét, viết nhận xét, truy xuất y văn và lắp ráp hai chương... (Quá trình này có thể mất từ 30-60 giây)"):
                        ch3_text, ch4_text = assemble_results_and_discussion_chapter(
                            selection_decisions=decisions,
                            saved_tables=saved_tabs,
                            chunks=chunks_db,
                            embeddings=embeddings_matrix,
                            bm25=bm25_index,
                            citation_engine=citation_eng,
                            study_context=study_ctx
                        )

                    st.session_state["assembled_ch3"] = ch3_text
                    st.session_state["assembled_ch4"] = ch4_text
                    st.success("🎉 Đã lắp ráp thành công toàn bộ bản thảo hai chương!")

            # Hiển thị kết quả nếu đã lắp ráp xong
            if st.session_state.get("assembled_ch3") and st.session_state.get("assembled_ch4"):
                tab_ch3, tab_ch4 = st.tabs(["📄 Chương 3: Kết quả", "📄 Chương 4: Bàn luận"])
                
                with tab_ch3:
                    st.markdown(st.session_state["assembled_ch3"])
                    docx_ch3 = create_word_document(title="Chương 3: Kết quả nghiên cứu", body=st.session_state["assembled_ch3"])
                    st.download_button("📥 Tải Chương 3 ra file Word", data=docx_ch3, file_name="Chuong_3_Ket_qua.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_ch3")

                with tab_ch4:
                    st.markdown(st.session_state["assembled_ch4"])
                    bib_text = citation_bibliography_wrapper()
                    docx_ch4 = create_word_document(title="Chương 4: Bàn luận", body=st.session_state["assembled_ch4"], bibliography=bib_text)
                    st.download_button("📥 Tải Chương 4 ra file Word (kèm TLTK)", data=docx_ch4, file_name="Chuong_4_Ban_luan.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_ch4")

            st.write("---")
            
            # ==========================================
            # 1. THỐNG KÊ MÔ TẢ (BIẾN PHÂN LOẠI)
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
            # 2. BIẾN ĐỊNH LƯỢNG (MEAN/SD, MEDIAN/IQR)
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
            # 3. BẢNG CHÉO & KIỂM ĐỊNH (CHI-SQUARE / FISHER / OR / CI95)
            # ==========================================
            st.subheader("3. Bảng chéo và kiểm định (Chi-square / Fisher / OR / CI95)")
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
                                        st.write(f"- **Chỉ số (Effect Size / OR):** `{result['effect_size']}`")

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
            # 4. SO SÁNH BIẾN ĐỊNH LƯỢNG GIỮA 2 NHÓM (T-TEST / MANN-WHITNEY)
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
                valid_num_cols = numeric_candidates if numeric_candidates else all_cols
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
            # 5. HỒI QUY LOGISTIC NHỊ PHÂN (OR VÀ 95% CI)
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
            # 8. DIỄN GIẢI KẾT QUẢ BẰNG AI (NHẬN XÉT BẢNG CHUYÊN SÂU)
            # ==========================================
            st.subheader("8. Diễn giải kết quả bằng AI (Nhận xét bảng chuẩn khoa học)")
            st.info("💡 Anh có thể chọn một bảng riêng lẻ hoặc chọn **'🌟 Chọn TẤT CẢ các bảng trong Giỏ'** để AI tổng hợp nhận xét toàn bộ số liệu.")

            saved_tables = st.session_state.get("saved_tables", {})
            table_options = ["-- Chỉ dùng số liệu dán tay bên dưới --", "🌟 Chọn TẤT CẢ các bảng trong Giỏ"] + list(saved_tables.keys())

            selected_table_key = st.selectbox(
                "Lựa chọn bảng hoặc nguồn dữ liệu:",
                options=table_options,
                key=ui_key("select_ai_table")
            )

            interpretation_request = st.text_area(
                "Số liệu bổ sung hoặc yêu cầu cụ thể (nếu có):",
                height=100,
                key=ui_key("interpretation_request")
            )

            if st.button("🤖 AI Viết Nhận Xét Bảng", type="primary", key="ai_interpret"):
                final_data = ""

                if selected_table_key == "🌟 Chọn TẤT CẢ các bảng trong Giỏ":
                    if not saved_tables:
                        st.warning("⚠️ Giỏ kết quả đang trống, chưa có bảng nào được lưu!")
                    else:
                        for t_key, df_t in saved_tables.items():
                            final_data += f"### BẢNG: {t_key}\n"
                            final_data += df_t.to_markdown(index=False) + "\n\n"
                elif selected_table_key != "-- Chỉ dùng số liệu dán tay bên dưới --":
                    df_target = saved_tables[selected_table_key]
                    final_data += f"### BẢNG: {selected_table_key}\n"
                    final_data += df_target.to_markdown(index=False) + "\n\n"

                if interpretation_request.strip():
                    final_data += f"SỐ LIỆU / YÊU CẦU BỔ SUNG:\n{interpretation_request.strip()}"

                if not final_data.strip():
                    st.warning("⚠️ Anh chưa chọn bảng nào hoặc chưa dán số liệu!")
                else:
                    try:
                        strict_remark_prompt = f"""
{BASE_SYSTEM_RULES}
Nhiệm vụ của bạn là viết phần **'Nhận xét'** cho các bảng số liệu thống kê trong luận văn Dược lâm sàng.
QUY TẮC VÀNG BẮT BUỘC:
1. CHỈ ĐƯA RA SỐ LIỆU: Chỉ diễn giải các số liệu, tần số, tỷ lệ % nổi bật (giá trị cao nhất, thấp nhất, trung vị, v.v.) có trong bảng.
2. VĂN PHONG KHOA HỌC: Câu văn logic, ngắn gọn, dễ hiểu, mạch lạc.
3. TUYỆT ĐỐI KHÔNG BÀN LUẬN: Không giải thích nguyên nhân, không suy diễn cơ chế lâm sàng, không so sánh với các nghiên cứu khác.
4. Trình bày thành các đoạn văn xuôi y khoa liền mạch, chuẩn mực.

DỮ LIỆU ĐẦU VÀO CẦN NHẬN XÉT:
{final_data}
"""

                        with st.spinner("AI đang phân tích số liệu và soạn nhận xét chuyên sâu..."):
                            output = call_gemini(strict_remark_prompt, model=DEFAULT_MODEL)
                            if output:
                                st.markdown("### 📝 Kết quả Nhận xét Bảng:")
                                st.markdown(output)
                    except Exception as exc:
                        st.error(f"Lỗi gọi AI: {exc}")
                        
    # ------------------------------------------------------------
    # TAB 5 – Audit
    # ------------------------------------------------------------
    with tabs[4]:
        st.header("🔎 Audit luận văn toàn diện")
        st.markdown('<div class="warning-box">⚠️ <b>Giới hạn cần biết:</b> Công cụ chỉ báo nguy cơ.</div>', unsafe_allow_html=True)
        Audit_text = st.text_area("Dán đoạn văn cần Audit vào đây:", height=250, key=ui_key("Audit_text"))
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        st.write("---")
        ket_qua_Audit_container = st.container()

        with c1:
            if st.button("🔢 Số liệu", use_container_width=True):
                if not Audit_text.strip(): st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang truy xuất và đối chiếu..."):
                        result = Audit_generated_text_wrapper(Audit_text)
                    with ket_qua_Audit_container:
                        st.markdown("### 🔢 Kết quả Audit Số liệu")
                        st.success(f"**Level 1 (Khớp chính xác):** {', '.join(result['exact_matches']) if result['exact_matches'] else 'Không có'}")
                        st.info(f"**Level 2 (Khớp phái sinh):** {', '.join(result['derived_matches']) if result['derived_matches'] else 'Không có'}")
                        if result["warnings"]: st.error(f"**Level 3 (⚠️ SỐ LIỆU LẠ):** {', '.join(result['warnings'])}")
                        else: st.success("**Level 3:** Không phát hiện số liệu lạ!")
                        with st.expander("📄 Xem bằng chứng đối chiếu"):
                            for ev in result["evidence_used"]: st.write(f"> {ev['text']}")

        with c2:
            if st.button("📚 Trích dẫn", use_container_width=True):
                if not Audit_text.strip(): st.warning("Chưa có văn bản.")
                else:
                    citations_in_text = re.findall(r"\[(\d+)\]", Audit_text)
                    current_refs = {str(ref['vancouver_index']): ref for ref in st.session_state.get("current_references", [])}
                    with ket_qua_Audit_container:
                        st.markdown("### 📚 Kết quả Audit Citation")
                        fake_citations = [c for c in citations_in_text if c not in current_refs]
                        if fake_citations: st.error(f"❌ Phát hiện trích dẫn ẢO: [{'], ['.join(fake_citations)}]")
                        elif citations_in_text: st.success("✅ Toàn bộ trích dẫn khớp!")
                        else: st.info("Không tìm thấy trích dẫn [n].")

        with c3:
            if st.button("🔍 Trùng lặp", use_container_width=True):
                if not Audit_text.strip(): st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang quét Jaccard Similarity..."):
                        overlaps = internal_overlap_Audit_wrapper(Audit_text)
                    with ket_qua_Audit_container:
                        st.markdown("### 🔍 Trùng lặp nội bộ")
                        if not overlaps: st.info("Không tìm thấy đoạn trùng đáng kể.")
                        else:
                            for item in overlaps: st.markdown(f"**{item['file']} – trang {item['page']}**\nSimilarity: **{item['similarity']}**\n> {item['text']}")

        with c4:
            if st.button("🔤 Chính tả", use_container_width=True):
                if not Audit_text.strip(): st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang rà soát lỗi chính tả..."):
                        prompt = f"{BASE_SYSTEM_RULES}\nRà soát đoạn văn bản sau để tìm lỗi chính tả/thuật ngữ. ĐOẠN VĂN: {Audit_text}"
                        res = call_gemini(prompt, model=MODEL_LITE)
                    with ket_qua_Audit_container: st.markdown("### 🔤 Chính tả & Thuật ngữ\n" + str(res))

        with c5:
            if st.button("🤖 Check văn AI", use_container_width=True):
                if not Audit_text.strip(): st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang phân tích dấu hiệu AI..."):
                        prompt = f"{BASE_SYSTEM_RULES}\nSoi khắt khe các dấu hiệu văn bản do AI viết. ĐOẠN VĂN: {Audit_text}"
                        res = call_gemini(prompt, model=MODEL_LITE)
                    with ket_qua_Audit_container: st.markdown("### 🤖 Chỉ báo nguy cơ AI viết\n" + str(res))

        with c6:
            if st.button("⚖️ Phản biện", use_container_width=True):
                if not Audit_text.strip(): st.warning("Chưa có văn bản.")
                else:
                    with st.spinner("Đang soi logic..."):
                        prompt = f"{BASE_SYSTEM_RULES}\nĐóng vai phản biện. Chỉ ra điểm yếu logic. ĐOẠN VĂN: {Audit_text}"
                        res = call_gemini(prompt)
                    with ket_qua_Audit_container: st.markdown("### ⚖️ Kết quả Phản biện\n" + str(res))   

    # ------------------------------------------------------------
    # TAB 6 – QUẢN LÝ TÀI LIỆU THAM KHẢO & METADATA (TLTK)
    # ------------------------------------------------------------
    with tabs[5]:
        st.header("🏷️ Tài liệu tham khảo")
        st.info("💡 Bảng dưới đây hiển thị thông tin thư mục của toàn bộ tài liệu anh đã nạp. Anh có thể **nhấp đúp chuột vào từng ô** để sửa thủ công tên tác giả, năm, tên bài... Nếu dữ liệu ở đây chuẩn, AI sẽ sinh ra danh mục Vancouver chuẩn xác.")
        
        docs = st.session_state.get("documents", {})
        
        if not docs:
            st.warning("⚠️ Chưa có tài liệu nào trong Evidence Database.")
        else:
            # 1. Hiển thị Data Editor cho phép sửa thủ công (Đã ép kiểu an toàn chống sập)
            doc_list = []
            for sid, meta in docs.items():
                doc_list.append({
                    "source_id": str(sid),
                    "origin": str(meta.get("origin") or "Khác"),
                    "authors": str(meta.get("authors") or ""),
                    "title": str(meta.get("title") or meta.get("file_name") or ""),
                    "journal": str(meta.get("journal") or ""),
                    "year": str(meta.get("year") or ""),
                    "doi": str(meta.get("doi") or "")
                })
            
            df_meta = pd.DataFrame(doc_list)
            
            edited_df = st.data_editor(
                df_meta,
                column_config={
                    "source_id": st.column_config.TextColumn("Mã ID", disabled=True),
                    "origin": st.column_config.TextColumn("Nguồn", disabled=True),
                    "authors": "Tác giả",
                    "title": "Tên bài báo / Tài liệu",
                    "journal": "Tạp chí / NXB",
                    "year": "Năm XB",
                    "doi": "DOI"
                },
                use_container_width=True,
                num_rows="fixed",
                key="meta_editor"
            )
            
            if st.button("💾 Lưu các chỉnh sửa bảng trên", type="primary"):
                # Cập nhật thay đổi từ bảng vào session_state
                for _, row in edited_df.iterrows():
                    sid = row["source_id"]
                    if sid in st.session_state["documents"]:
                        st.session_state["documents"][sid]["authors"] = row["authors"]
                        st.session_state["documents"][sid]["title"] = row["title"]
                        st.session_state["documents"][sid]["journal"] = row["journal"]
                        st.session_state["documents"][sid]["year"] = str(row["year"]) if pd.notna(row["year"]) else ""
                        st.session_state["documents"][sid]["doi"] = row["doi"]
                st.success("✅ Đã cập nhật thông tin thư mục thành công! AI sẽ sử dụng thông tin này để tạo trích dẫn Vancouver.")
                time.sleep(1)
                st.rerun()

            st.write("---")
            
            # Đã xóa phần chia cột và nút AI quét lại, chỉ giữ lại phần hiển thị Vancouver tràn viền
            st.subheader("📖 Danh sách Vancouver hiện tại")
            st.caption("Đây là danh sách trích dẫn đã được sử dụng trong bản nháp (Tab 3).")
            registry = st.session_state.get("citation_registry", {})
            if registry:
                bib = citation_bibliography_wrapper()
                st.code(bib if bib else "Chưa có trích dẫn.", language="text")
            else:
                st.info("Chưa có trích dẫn nào được sinh ra trong bản nháp.")


if __name__ == "__main__":
    main()
