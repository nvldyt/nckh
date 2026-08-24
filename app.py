# app.py
# ============================================================
# HỖ TRỢ NGHIÊN CỨU KHOA HỌC – EVIDENCE-BASED RAG
# Bản tối ưu cho luận văn Chuyên khoa cấp I – Dược lâm sàng
# Kiến trúc Modular 4 Engines (Tích hợp Reranker & Study Context)
# Bổ sung tính năng Offline: Summarizer (Python) & Ollama Writer
# ============================================================

import io
import os
import re
import time
import json
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from docx import Document

from chat_data_engine import render_chat_assistant
from chat_writing_engine import render_writing_chat

from table_selection_engine import (
    StudyObjective, CandidateResult, TableSelectionEngine,
    NarrativePlanner, Priority, Presentation,
)
from statistical_engine import (
    validate_dataframe, descriptive_table, numeric_summary,
    crosstab_test, compare_two_groups, binary_logistic_regression,
    create_clinical_groups, generate_baseline_table,
)
from evidence_engine import (
    SourceDocument, EvidenceChunk, extract_pdf,
    search_pubmed, ingest_pubmed_article, 
    add_source_and_chunks
)
from project_storage import save_project, load_project, list_projects, delete_project
from data_engine import auto_clean_data

from citation_engine import CitationEngine
from audit_engine import Audit_generated_text, internal_overlap_Audit
from retrieval_engine import (
    build_bm25_index, build_embedding_index,
    update_embedding_index, retrieve_evidence,
)
from writing_engine import (
    call_gemini, generate_evidence_based,
    BASE_SYSTEM_RULES, MODEL_LITE, DEFAULT_MODEL,
)
from synthesis_engine import build_literature_matrix
from chapter_assembler_engine import assemble_results_and_discussion_chapter
from offline_writing import render_offline_tab

# --- IMPORT 2 MODULE MỚI ---
from summarizer_engine import render_summarizer_tab
from ollama_writer_engine import render_ollama_writer_tab


# ============================================================
# 1. CẤU HÌNH
# ============================================================
st.set_page_config(page_title="NCKH", page_icon="🔬", layout="wide")

DEFAULT_TOP_K = 8
MAX_TOP_K = 20
DEFAULT_VN_JOURNAL_DOMAINS = [
    "tapchiyhocvietnam.vn", "vjol.info", "tapchinghiencuuyhoc.vn",
    "jmp.huemed-univ.edu.vn", "jmpm.vn", "huejmp.vn",
    "tcydls108.benhvien108.vn", "tapchiyhcd.vn", "thaibinhjmp.vn", "hup.edu.vn",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,p,h1,h2,h3,h4,h5,h6,span,div,label,li,.stMarkdown{font-family:'Inter',sans-serif}
.stApp{background:#f8f9fa;color:#2c3e50}
h1{color:#1e293b!important;text-align:center;font-weight:800;margin-top:0!important;margin-bottom:6px}
h2,h3,h4{color:#0f172a!important;font-weight:600}
.stTabs [data-baseweb="tab-panel"]{background:#fff;border-radius:12px;padding:32px;box-shadow:0 1px 3px rgba(0,0,0,.05);border:1px solid #e2e8f0;margin-top:10px}
.stTabs [data-baseweb="tab-list"]{background:#f1f5f9;border-radius:10px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;font-weight:600;color:#64748b}
.stTabs [aria-selected="true"]{background:#fff!important;color:#2563eb!important;box-shadow:0 1px 3px rgba(0,0,0,.1)}
div.stButton>button,div.stDownloadButton>button{background:#fff!important;color:#334155!important;font-weight:600;border-radius:8px;border:1px solid #cbd5e1!important;padding:8px 16px}
div.stButton>button[kind="primary"]{background:#2563eb!important;color:#fff!important;border:none!important}
.warning-box{border-left:4px solid #f59e0b;padding:12px 16px;background:#fffbeb;border-radius:6px;color:#92400e;font-size:.95rem}
.danger-box{border-left:4px solid #ef4444;padding:12px 16px;background:#fef2f2;border-radius:6px;color:#991b1b;font-size:.95rem}
.success-box{border-left:4px solid #10b981;padding:12px 16px;background:#ecfdf5;border-radius:6px;color:#065f46;font-size:.95rem}
</style>
""", unsafe_allow_html=True)

UI_NAMESPACE = "nckh_cki"


def init_state():
    if "ui_version" not in st.session_state:
        st.session_state["ui_version"] = 0
    defaults = {
        "documents": {}, "chunks": [], "embeddings": None, "bm25": None,
        "citation_registry": {}, "Audit_log": [], "last_generated": "",
        "last_evidence": [], "current_references": [],
        "vn_journal_domains": list(DEFAULT_VN_JOURNAL_DOMAINS),
        "t3_pm_data": [], "t3_vn_data": [], "t3_query": "",
        "t3_en_keyword": "", "t3_vn_keyword": "",
        "result_cart": [], "saved_tables": {},
        "selection_decisions": [], "narrative_plan": {},
        "study_context": {}, "literature_matrix_df": pd.DataFrame(),
        "assembled_ch3": "", "assembled_ch4": "",
        "excel_data": None, "excel_df": None, "clean_logs": [],
        "ai_pending_remark": "",
        "cached_summary": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (list, dict, pd.DataFrame)) else value


def ui_key(widget_name: str) -> str:
    return f"{UI_NAMESPACE}_v{st.session_state.get('ui_version',0)}_{widget_name}"


def reset_ui_state():
    st.session_state["ui_version"] = st.session_state.get("ui_version",0) + 1


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


# ============================================================
# 2. WRAPPERS – KẾT NỐI UI VỚI ENGINES
# ============================================================
def rebuild_index(new_chunks: Optional[List[Dict[str, Any]]] = None):
    all_chunks = st.session_state.get("chunks", [])
    if not all_chunks:
        st.session_state["embeddings"] = None
        st.session_state["bm25"] = None
        gc.collect()
        return
    old = st.session_state.get("embeddings")
    try:
        if new_chunks and old is not None:
            st.session_state["embeddings"] = update_embedding_index(new_chunks, old)
        else:
            st.session_state["embeddings"] = build_embedding_index(all_chunks)
        del old
        gc.collect()
        st.session_state["bm25"] = build_bm25_index(all_chunks)
    except Exception:
        gc.collect()
        raise


def retrieve_evidence_wrapper(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    return retrieve_evidence(
        query=query,
        chunks=st.session_state.get("chunks", []),
        matrix=st.session_state.get("embeddings"),
        bm25=st.session_state.get("bm25"),
        top_k=k,
    )


def get_citation_engine() -> CitationEngine:
    if "citation_engine" not in st.session_state:
        st.session_state["citation_engine"] = CitationEngine()
    return st.session_state["citation_engine"]


def source_metadata(source_id: str) -> Dict[str, Any]:
    return st.session_state.get("documents", {}).get(source_id, {})


def citation_bibliography_wrapper() -> str:
    rows = []
    docs = st.session_state.get("documents", {})
    for ref in st.session_state.get("current_references", []):
        rid = ref.get("ref_id", "")
        sid = rid.replace("REF-", "", 1) if rid.startswith("REF-") else rid
        meta = docs.get(sid, ref.get("metadata", {})) or {}
        authors = meta.get("authors") or "Tác giả chưa xác định"
        title = meta.get("title") or meta.get("file_name") or "Tài liệu chưa xác định"
        journal = meta.get("journal") or "Tài liệu lưu trữ"
        year = meta.get("year") or "Năm chưa rõ"
        text = f"[{ref.get('vancouver_index','')}] {authors}. {title}. {journal}. {year}."
        if meta.get("doi"):
            text += f" DOI: {meta['doi']}."
        if meta.get("pmid"):
            text += f" PMID: {meta['pmid']}."
        if meta.get("url") and meta.get("origin") == "Tạp chí VN":
            text += f" [{meta['url']}]"
        rows.append(text)
    return "\n".join(rows)


def generate_evidence_based_wrapper(task: str, query: str, k: int = DEFAULT_TOP_K):
    evidence = retrieve_evidence_wrapper(query, k=k)
    if not evidence:
        return "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận.", [], []
    engine = get_citation_engine()
    final_text, references, invalid_tags = generate_evidence_based(
        task_prompt=task,
        evidence=evidence,
        citation_engine=engine,
        study_context=st.session_state.get("study_context", {}),
    )
    if final_text:
        st.session_state["last_generated"] = final_text
        st.session_state["last_evidence"] = evidence
        st.session_state["current_references"] = references
    return final_text, evidence, invalid_tags


def Audit_generated_text_wrapper(text: str):
    return Audit_generated_text(text, retrieve_evidence_wrapper(text, k=6))


def internal_overlap_Audit_wrapper(text: str, top_k: int = 5):
    return internal_overlap_Audit(text, st.session_state.get("chunks", []), top_k=top_k)


def extract_metadata_from_text_ai_wrapper(text: str) -> dict:
    prompt = f"""
Bạn là chuyên gia thư viện y khoa. Nhiệm vụ của bạn là trích xuất siêu dữ liệu (metadata) từ văn bản thô của trang đầu tiên của một bài báo nghiên cứu.

ĐẶC BIỆT LƯU Ý VỚI BÀI BÁO TIẾNG VIỆT:
1. Tác giả (authors): Thường nằm ngay dưới tiêu đề bài báo. Hãy lọc bỏ tên cơ quan/bệnh viện/đại học. Gom các tên người lại, cách nhau bằng dấu phẩy.
2. Tạp chí (journal): Tìm các cụm từ bắt đầu bằng "Tạp chí", "Y học", "Nghiên cứu", "Y dược", "Journal".
3. Năm xuất bản (year): Tìm con số 4 chữ số hợp lý nhất.

TRẢ VỀ DUY NHẤT MỘT CHUỖI JSON HỢP LỆ, KHÔNG GIẢI THÍCH GÌ THÊM.
Cấu trúc JSON bắt buộc:
{{"authors":"...","title":"...","year":"...","journal":"...","doi":"..."}}

ĐOẠN VĂN BẢN QUÉT ĐƯỢC:
{text[:4500]}
"""
    try:
        res = call_gemini(prompt, model=DEFAULT_MODEL, temperature=0.0)
    except Exception:
        return {}
    if not res:
        return {}
    try:
        cleaned = res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        out = json.loads(cleaned.strip())
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
def add_pdf_documents(uploaded_files) -> Tuple[int, int, List[str]]:
    new_sources = 0
    new_chunks_count = 0
    errors = []
    new_chunks_list = []
    for uploaded_file in uploaded_files:
        try:
            source, chunks = extract_pdf(uploaded_file)
            if add_source_and_chunks(source, chunks):
                new_sources += 1
                new_chunks_count += len(chunks)
                new_chunks_list.extend(chunks)
                sid = _field(source, "source_id")
                if sid and chunks:
                    page_one = []
                    for c in chunks:
                        page = str(_field(c, "page", "") or "").strip().lower()
                        txt = _field(c, "text", "") or ""
                        if page in {"1", "trang 1", "page 1"}:
                            page_one.append(txt)
                    target = page_one[0] if page_one else (_field(chunks[0], "text", "") or "")
                    if target:
                        meta = extract_metadata_from_text_ai_wrapper(target)
                        if meta and sid in st.session_state.get("documents", {}):
                            for k, v in meta.items():
                                if v:
                                    st.session_state["documents"][sid][k] = v
                    gc.collect()
        except Exception as exc:
            errors.append(f"{getattr(uploaded_file,'name','PDF')}: {exc}")
            gc.collect()
    if new_sources:
        rebuild_index(new_chunks=new_chunks_list)
    del new_chunks_list
    gc.collect()
    return new_sources, new_chunks_count, errors


def evidence_database_summary() -> Dict[str, Any]:
    docs = st.session_state.get("documents", {})
    chunks = st.session_state.get("chunks", [])
    by_src = {}
    for meta in docs.values():
        o = meta.get("origin", "Khác")
        by_src[o] = by_src.get(o, 0) + 1
    sid_origin = {sid: meta.get("origin", "Khác") for sid, meta in docs.items()}
    by_chunk = {}
    for c in chunks:
        o = sid_origin.get(_field(c, "source_id"), "Khác")
        by_chunk[o] = by_chunk.get(o, 0) + 1
    return {
        "total_sources": len(docs), "total_chunks": len(chunks),
        "by_origin_sources": by_src, "by_origin_chunks": by_chunk,
        "index_ready": st.session_state.get("embeddings") is not None,
    }


def render_evidence_database_status(context_label: str = ""):
    s = evidence_database_summary()
    if not s["total_sources"]:
        st.markdown('<div class="danger-box">⚠️ <b>Evidence Database đang RỖNG.</b> Hãy nạp PDF ở Tab 1 hoặc tra cứu và nạp bài báo ở Tab 2.</div>', unsafe_allow_html=True)
        return
    labels = {"PDF":"📄 PDF","PubMed":"🌍 PubMed","Tạp chí VN":"🇻🇳 Tạp chí VN","Khác":"❓ Khác"}
    pieces = []
    for origin, count in s["by_origin_sources"].items():
        pieces.append(f"{labels.get(origin, origin)}: <b>{count}</b> nguồn ({s['by_origin_chunks'].get(origin,0)} đoạn)")
    st.markdown(
        f'<div class="success-box">✅ <b>Evidence Database{(" - "+context_label) if context_label else ""}:</b> {s["total_sources"]} nguồn / {s["total_chunks"]} đoạn &nbsp;—&nbsp; {" &nbsp;|&nbsp; ".join(pieces)}{("" if s["index_ready"] else " &nbsp;—&nbsp; ⚠️ chưa dựng xong index")}</div>',
        unsafe_allow_html=True,
    )


def translate_query_to_mesh(vietnamese_query: str) -> str:
    prompt = f"""Bạn là một chuyên gia tra cứu tài liệu y khoa.
Chuyển tên đề tài tiếng Việt sau thành chuỗi từ khóa tiếng Anh hiệu quả để tra cứu PubMed.
Dùng AND, OR; không dùng dấu gạch chéo; giữ 2-4 cụm từ khóa; chỉ trả về chuỗi tiếng Anh.
Đề tài: {vietnamese_query}"""
    try:
        text = call_gemini(prompt, model=MODEL_LITE)
    except Exception:
        return vietnamese_query
    return text.strip().strip('"').strip("'").replace("\n", " ") if text else vietnamese_query


def normalize_pubmed_query(en_query: str, fallback_query: str) -> str:
    q = (en_query or "").split("###")[0].split("---")[0].strip()
    markers = ["hoặc", "nếu", "về", "trong", "đề tài", "từ khóa", "từ khoá"]
    if not q or any(x in q.lower() for x in markers):
        return fallback_query.strip()
    return q


def add_markdown_body_to_doc(doc: Document, body: str):
    lines = body.split("\n")
    buffer = []
    def flush():
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                doc.add_paragraph(text)
            buffer.clear()
    for line in lines:
        s = line.strip()
        is_table = s.startswith("|")
        if not is_table:
            doc._last_table_open = False
        if not s:
            flush(); continue
        if s.startswith("### "):
            flush(); doc.add_heading(s[4:].strip(), level=3)
        elif s.startswith("## "):
            flush(); doc.add_heading(s[3:].strip(), level=2)
        elif s.startswith("# "):
            flush(); doc.add_heading(s[2:].strip(), level=1)
        elif s.startswith(("- ", "* ")):
            flush(); doc.add_paragraph(s[2:].strip(), style="List Bullet")
        elif is_table:
            test = set(s.replace("|","").replace(" ","").replace(":",""))
            if test and test <= {"-"}:
                continue
            flush()
            cells = [x.strip() for x in s.strip("|").split("|")]
            table = doc.tables[-1] if doc.tables and getattr(doc,"_last_table_open",False) else None
            if table is None:
                table = doc.add_table(rows=1, cols=max(1,len(cells)))
                table.style = "Light Grid Accent 1"
                for i, cell in enumerate(cells): table.rows[0].cells[i].text = cell
                doc._last_table_open = True
            else:
                row = table.add_row()
                for i, cell in enumerate(cells):
                    if i < len(row.cells): row.cells[i].text = cell
        else:
            buffer.append(s)
    flush()


def create_word_document(title: str, body: str, bibliography: str = "") -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)
    add_markdown_body_to_doc(doc, body)
    if bibliography.strip():
        doc.add_heading("Tài liệu tham khảo", level=1)
        for item in bibliography.splitlines():
            if item.strip(): doc.add_paragraph(item.strip())
    out = io.BytesIO(); doc.save(out); return out.getvalue()


def safe_read_excel(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.astype(str)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)
        gc.collect()
        return df
    except Exception as exc:
        st.error(f"❌ Lỗi khi đọc Excel: {exc}")
        return pd.DataFrame()


def large_batch_ok(name: str, count: int, threshold: int = 250) -> bool:
    if count <= threshold:
        return True
    st.warning(f"⚠️ {name} sẽ xử lý khoảng **{count} tổ hợp** và có thể dùng nhiều RAM.")
    return st.checkbox("Tôi hiểu và muốn chạy toàn bộ tổ hợp này", key=ui_key(f"confirm_{name}_{count}"))


def upsert_candidate(result_id, title, result_type, variables, scientific_value, clinical_importance, discussion_value, p_value=None):
    cart = st.session_state.setdefault("result_cart", [])
    cart[:] = [x for x in cart if getattr(x,"id",None) != result_id]
    cart.append(CandidateResult(
        id=result_id, title=title, result_type=result_type,
        variables=variables, p_value=p_value,
        scientific_value=scientific_value,
        clinical_importance=clinical_importance,
        discussion_value=discussion_value,
    ))


def project_payload() -> dict:
    return {
        "documents": st.session_state.get("documents", {}),
        "chunks": st.session_state.get("chunks", []),
        "citation_registry": st.session_state.get("citation_registry", {}),
        "current_references": st.session_state.get("current_references", []),
        "result_cart": [vars(x) if hasattr(x,"__dict__") else x for x in st.session_state.get("result_cart",[])],
        "saved_tables": {k:v.to_dict(orient="split") for k,v in st.session_state.get("saved_tables",{}).items()},
        "study_context": st.session_state.get("study_context", {}),
        "narrative_plan": st.session_state.get("narrative_plan", {}),
        "assembled_ch3": st.session_state.get("assembled_ch3", ""),
        "assembled_ch4": st.session_state.get("assembled_ch4", ""),
        "vn_journal_domains": st.session_state.get("vn_journal_domains", list(DEFAULT_VN_JOURNAL_DOMAINS)),
        "cached_summary": st.session_state.get("cached_summary", ""),
    }


def restore_project(data: dict):
    st.session_state["documents"] = data.get("documents", {})
    st.session_state["chunks"] = data.get("chunks", [])
    st.session_state["citation_registry"] = data.get("citation_registry", {})
    st.session_state["current_references"] = data.get("current_references", [])
    st.session_state["study_context"] = data.get("study_context", {})
    st.session_state["narrative_plan"] = data.get("narrative_plan", {})
    st.session_state["assembled_ch3"] = data.get("assembled_ch3", "")
    st.session_state["assembled_ch4"] = data.get("assembled_ch4", "")
    st.session_state["vn_journal_domains"] = data.get("vn_journal_domains", list(DEFAULT_VN_JOURNAL_DOMAINS))
    st.session_state["cached_summary"] = data.get("cached_summary", "")
    st.session_state["result_cart"] = []
    for item in data.get("result_cart", []):
        if isinstance(item, dict):
            try: st.session_state["result_cart"].append(CandidateResult(**item))
            except Exception: pass
        else:
            st.session_state["result_cart"].append(item)
    st.session_state["saved_tables"] = {}
    for k,v in data.get("saved_tables",{}).items():
        try: st.session_state["saved_tables"][k] = pd.DataFrame.from_dict(v, orient="split")
        except Exception: pass
    rebuild_index()
    gc.collect()


# ============================================================
# 4. MAIN UI - 9 TABS TÍCH HỢP ĐẦY ĐỦ
# ============================================================
def main():
    init_state()
    st.title("🔬 HỖ TRỢ NGHIÊN CỨU KHOA HỌC")
    st.caption("Evidence-Based RAG • PubMed • Citation Registry • Statistical Engine • Audit")

    with st.sidebar:
        st.header("⚙️ Quản lý dự án")
        if st.button("🧹 Làm mới Giao diện", use_container_width=True):
            reset_ui_state(); st.rerun()
        st.divider(); st.subheader("💾 Lưu & Khôi phục (.json)")
        pdata = json.dumps(project_payload(), ensure_ascii=False, indent=4)
        st.download_button("📥 Tải file dự án", data=pdata, file_name="Du_An_Luan_Van.json", mime="application/json", use_container_width=True)
        uploaded_proj = st.file_uploader("Khôi phục từ file:", type=["json"], key=ui_key("upload_project_json"))
        if uploaded_proj and st.button("🚀 Khôi phục", type="primary", use_container_width=True):
            try:
                restore_project(json.load(uploaded_proj)); st.success("🎉 Khôi phục thành công!"); time.sleep(.8); reset_ui_state(); st.rerun()
            except Exception as exc:
                st.error(f"❌ Lỗi: {exc}")
    
    tabs = st.tabs([
        "🔍 1. PubMed", 
        "📑 2. Tài liệu",
        "🛠️ 3. Tổng hợp", 
        "⚡ 4. Tóm tắt",
        "✍️ 5. Viết văn (Grok AI)", 
        "🤖 6. Viết văn (Ollama)",
        "📊 7. Phân tích số liệu", 
        "🔎 8. Kiểm tra luận văn", 
        "🏷️ 9. Trích dẫn TLTK",
    ])

    # ------------------------------------------------------------
    # TAB 1 - PubMed
    # ------------------------------------------------------------
    with tabs[0]:
        st.header("🔍 PubMed")
        st.info("Nhập tên đề tài bằng tiếng Việt. Hệ thống tự dịch sang từ khoá MeSH để tra cứu tài liệu y khoa chuẩn quốc tế trên PubMed.")
        render_evidence_database_status()
        
        cs, cb = st.columns([4, 1])
        with cs: 
            t3_query = st.text_input("Tên đề tài nghiên cứu (tiếng Việt):", key=ui_key("t3_query_input"))
        with cb: 
            # ĐÃ MỞ KHÓA: cho phép lấy tối đa 50 bài, mặc định là 20
            max_res = st.number_input("Số bài/nguồn", min_value=2, max_value=50, value=20, key=ui_key("t3_max_res"))
            
        if st.button("🚀 Tìm kiếm trên PubMed", type="primary", key=ui_key("t3_btn_search")):
            if not t3_query.strip(): 
                st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
            else:
                st.session_state["t3_query"] = t3_query
                with st.spinner("Đang dịch & chuẩn hoá từ khoá sang MeSH (PubMed)..."):
                    en_query = translate_query_to_mesh(t3_query)
                    st.session_state["t3_en_keyword"] = en_query
                    
                with st.spinner(f"Đang tìm & tải {max_res} Abstract từ PubMed..."):
                    st.session_state["t3_pm_data"] = search_pubmed(normalize_pubmed_query(en_query, t3_query), max_res)
                    
        # HIỂN THỊ KẾT QUẢ PUBMED
        if st.session_state.get("t3_pm_data"):
            st.write("---")
            st.markdown("### 🌍 Kết quả tra cứu PubMed")
            if st.session_state.get("t3_en_keyword"): 
                st.success(f"🔑 Từ khoá MeSH: **{st.session_state['t3_en_keyword']}**")
                
            for i, art in enumerate(st.session_state.get("t3_pm_data", [])):
                with st.container(border=True):
                    st.markdown(f"**[{art.get('title', '')}]({art.get('url', '#')})**")
                    st.caption(f"✍️ {art.get('authors', '')} ({art.get('year', '')}) — {art.get('journal', '')}")
                    with st.expander("Xem tóm tắt (Abstract)"): 
                        st.write(art.get("abstract", ""))
                        
                    if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"pm_ingest_{i}")):
                        try:
                            if ingest_pubmed_article(art): 
                                rebuild_index()
                                st.success("Đã nạp vào Evidence Database.")
                                st.rerun()
                            else: 
                                st.info("Nguồn này đã có trong Evidence Database.")
                        except Exception as exc: 
                            st.error(f"❌ Lỗi nạp nguồn: {exc}")
                            
            st.write("---")
            if st.button("➕ Nạp TẤT CẢ kết quả ở trên vào Evidence Database", key=ui_key("t3_ingest_all")):
                count = 0
                with st.spinner("Đang nạp tất cả nguồn và cập nhật index..."):
                    for art in st.session_state.get("t3_pm_data", []):
                        try: 
                            count += 1 if ingest_pubmed_article(art) else 0
                        except Exception: 
                            pass
                    if count: 
                        rebuild_index()
                st.success(f"Đã nạp {count} nguồn mới vào Evidence Database.")

    # ------------------------------------------------------------
    # TAB 2 - Tài liệu (PDF)
    # ------------------------------------------------------------
    with tabs[1]:
        st.header("📚 Ngân hàng tài liệu gốc (PDF)")
        render_evidence_database_status()
        uploaded_files = st.file_uploader("Tải PDF nghiên cứu / guideline / bài báo", type=["pdf"], accept_multiple_files=True, key=ui_key("pdf_uploader"))
        c1,c2=st.columns(2)
        with c1:
            if st.button("📥 Nạp tài liệu vào Evidence Database", type="primary"):
                if not uploaded_files: st.warning("Chưa có file PDF.")
                else:
                    with st.spinner("Đang đọc PDF, tạo source registry và embedding..."):
                        ns,nc,errors=add_pdf_documents(uploaded_files)
                    st.success(f"Đã thêm {ns} tài liệu, {nc} phân đoạn bằng chứng.")
                    for e in errors: st.error(e)
        with c2:
            if st.button("🗑️ Xóa toàn bộ ngân hàng tài liệu"):
                for k,v in {"documents":{},"chunks":[],"embeddings":None,"bm25":None,"citation_registry":{},"last_evidence":[]}.items(): st.session_state[k]=v
                gc.collect(); st.success("Đã xóa dữ liệu trong phiên hiện tại."); st.rerun()
        st.write("---"); st.subheader("Nguồn PDF đã nạp")
        docs=list(st.session_state.get("documents",{}).values())
        if docs: st.dataframe(pd.DataFrame(docs), use_container_width=True)
        else: st.info("Chưa có tài liệu.")
        st.subheader("Tìm bằng chứng trong toàn bộ Evidence Database")
        evidence_query=st.text_area("Nhập vấn đề cần tìm trong tài liệu:",placeholder="Ví dụ: tỷ lệ bệnh nhân đạt huyết áp mục tiêu...",key=ui_key("evidence_query"))
        top_k=st.slider("Số đoạn bằng chứng",3,MAX_TOP_K,DEFAULT_TOP_K,key=ui_key("top_k_tab1"))
        if st.button("🔎 Truy xuất bằng chứng",key=ui_key("retrieve_tab1")):
            if not evidence_query.strip(): st.warning("Nhập câu hỏi trước.")
            else:
                evidence=retrieve_evidence_wrapper(evidence_query,k=top_k); st.session_state["last_evidence"]=evidence
                if not evidence: st.warning("Không tìm thấy bằng chứng trong tài liệu.")
                else:
                    for ev in evidence:
                        meta=st.session_state["documents"].get(ev.get("source_id"),{})
                        st.markdown(f"**{ev.get('chunk_id','')}** _({meta.get('origin','')})_\nNguồn: {ev.get('file_name','')} — Trang/mục: {ev.get('page','')}\nĐiểm: {ev.get('score',0):.4f}\n\n> {ev.get('text','')}")

    # ------------------------------------------------------------
    # TAB 3 – TỔNG HỢP OFFLINE
    # ------------------------------------------------------------
    with tabs[2]:
        render_offline_tab()

    # ------------------------------------------------------------
    # TAB 4 – TÓM TẮT BẰNG PYTHON
    # ------------------------------------------------------------
    with tabs[3]:
        render_summarizer_tab()

    # ------------------------------------------------------------
    # TAB 5 – VIẾT LUẬN VĂN (GEMINI API)
    # ------------------------------------------------------------
    with tabs[4]:
        st.header("📝 Viết tự động bằng AI (RAG)")
        st.header("✍️ Viết luận văn dựa trên bằng chứng")
        st.warning("Đây là công cụ tạo bản nháp. Mọi citation và số liệu phải được Audit lại (đối chiếu bản gốc) trước khi đưa vào luận văn chính thức.")
        render_evidence_database_status("dùng cho các nút viết nhanh bên dưới")

        with st.expander("🎯 KHAI BÁO BỐI CẢNH NGHIÊN CỨU (STUDY CONTEXT) - Cấu hình 1 lần dùng mãi mãi",expanded=True):
            st.info("💡 Khai báo thông tin đề tài tại đây để AI tự động hiểu và bám sát vào mục tiêu của anh trong mọi lần sinh văn bản, không sợ lạc đề.")
            ctx=st.session_state.get("study_context",{}); a,b=st.columns(2)
            with a:
                ctx_title=st.text_input("Tên đề tài:",value=ctx.get("title",""),placeholder="VD: Phân tích tình hình sử dụng thuốc...",key=ui_key("ctx_title"))
                ctx_design=st.text_input("Thiết kế nghiên cứu:",value=ctx.get("design",""),placeholder="VD: Mô tả cắt ngang hồi cứu",key=ui_key("ctx_design"))
                ctx_population=st.text_input("Đối tượng bệnh nhân:",value=ctx.get("population",""),placeholder="VD: Bệnh nhân tăng huyết áp ngoại trú",key=ui_key("ctx_population"))
            with b:
                ctx_sample=st.text_input("Cỡ mẫu dự kiến (N=):",value=ctx.get("sample_size",""),placeholder="VD: 150 bệnh án",key=ui_key("ctx_sample"))
                ctx_obj=st.text_area("Mục tiêu chính:",value=ctx.get("objectives",""),height=110,placeholder="VD: 1. Khảo sát đặc điểm... 2. Đánh giá tính hợp lý...",key=ui_key("ctx_objectives"))
            if st.button("💾 Lưu Study Context",key=ui_key("save_study_context")):
                st.session_state["study_context"]={"title":ctx_title,"design":ctx_design,"population":ctx_population,"sample_size":ctx_sample,"objectives":ctx_obj}; st.success("✅ Đã lưu bối cảnh! Bộ não AI đã được đồng bộ hóa với đề tài của anh.")

        with st.expander("🌟 Tự động lập Ma trận tổng hợp y văn từ Evidence Database",expanded=False):
            st.info("💡 Tính năng này tự động quét tất cả các tài liệu / bài báo bạn đã nạp, tổng hợp thành bảng so sánh chuẩn y khoa (Tác giả, Năm, Thiết kế, Cỡ mẫu, Kết quả chính).")
            if st.button("🚀 Khởi tạo Ma trận Tổng hợp Y văn",type="primary",key=ui_key("btn_build_matrix")):
                docs=st.session_state.get("documents",{}); chunks=st.session_state.get("chunks",[])
                if not docs: st.warning("⚠️ Evidence Database đang trống! Hãy nạp tài liệu PDF hoặc bài báo từ PubMed/Tạp chí VN trước.")
                else:
                    with st.spinner("AI đang phân tích và cấu trúc hóa ma trận y văn..."):
                        try: matrix_df=build_literature_matrix(docs,chunks)
                        except Exception as exc: matrix_df=pd.DataFrame(); st.error(f"❌ Lỗi khi lập ma trận: {exc}")
                    if not matrix_df.empty:
                        st.session_state["literature_matrix_df"]=matrix_df; st.success("✅ Đã lập thành công Ma trận tổng hợp y văn!"); st.dataframe(matrix_df,use_container_width=True)
                        mb=create_word_document("Ma trận Tổng hợp Y văn","### Ma trận tổng hợp y văn\n\n"+matrix_df.to_markdown(index=False))
                        st.download_button("📥 Tải Ma trận Y văn ra file Word",data=mb,file_name="Ma_tran_Tong_hop_Y_van.docx",mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",key=ui_key("download_matrix_word"))
                    else: st.error("❌ Không thể trích xuất dữ liệu ma trận. Vui lòng thử lại.")

        st.markdown("### 📊 Dữ liệu nghiên cứu của riêng anh")
        if st.session_state.get("ai_pending_remark"):
            st.session_state[ui_key("my_table_remarks")]=st.session_state["ai_pending_remark"]; st.session_state["ai_pending_remark"]=""
        c1,c2=st.columns(2)
        with c1: my_research_data=st.text_area("1. Số liệu bảng (Dán bảng Excel/Markdown vào đây):",height=180,key=ui_key("my_research_data"))
        with c2: my_table_remarks=st.text_area("2. Nhận xét bảng (Chỉ diễn giải số liệu, KHÔNG bàn luận):",height=180,key=ui_key("my_table_remarks"))
        if st.button("✍️ AI Viết Nhận Xét Bảng (Tự động điền vào ô 2)",key=ui_key("write_table_remark")):
            if not my_research_data.strip(): st.warning("⚠️ Anh cần dán bảng số liệu vào ô số 1 trước để AI có dữ liệu đọc!")
            else:
                prompt=f"{BASE_SYSTEM_RULES}\nNHIỆM VỤ:\nNgắn gọn, CHỈ diễn giải các số liệu nổi bật. KHÔNG bàn luận, KHÔNG so sánh. Viết thành MỘT ĐOẠN VĂN LIỀN MẠCH duy nhất.\nBẢNG SỐ LIỆU:\n{my_research_data}"
                try:
                    with st.spinner("AI đang phân tích bảng và soạn nhận xét..."): generated_remark=call_gemini(prompt,model=DEFAULT_MODEL)
                except Exception as exc: generated_remark=None; st.error(f"❌ Lỗi gọi Gemini: {exc}")
                if generated_remark: st.session_state["ai_pending_remark"]=generated_remark; st.rerun()

        citation_rules="Chỉ dùng SOURCE_TAG thật để hệ thống tự chuyển thành [n]. Các số trích dẫn theo thứ tự xuất hiện."
        result_box=st.container()

        # --- THÊM HÀM BỘ LỌC VÀO TRƯỚC RUN_QUICK_TASK ---
        import re
        def format_numbered_citations(generated_text: str) -> str:
            pattern = r'\[(?:REF-)?(SRC-[A-Z0-9]+)\]'
            citation_mapping = {}
            current_index = 1
            
            def replacer(match):
                nonlocal current_index
                ref_id = match.group(1)
                if ref_id not in citation_mapping:
                    citation_mapping[ref_id] = current_index
                    current_index += 1
                return f"[{citation_mapping[ref_id]}]"
                
            clean_text = re.sub(pattern, replacer, generated_text)
            
            # --- PHẦN BỔ SUNG: ĐỒNG BỘ DANH SÁCH VỚI TAB 9 ---
            new_refs = []
            docs = st.session_state.get("documents", {})
            for ref_id, idx in citation_mapping.items():
                new_refs.append({
                    "ref_id": ref_id,
                    "vancouver_index": idx,
                    "metadata": docs.get(ref_id, {})
                })
            
            # Ép hệ thống lưu danh sách chuẩn vào bộ nhớ để Tab 9 & lúc xuất file Word có thể đọc được
            st.session_state["current_references"] = new_refs
            st.session_state["citation_registry"] = citation_mapping
            
            return clean_text
        
        import re
        import re

        def run_quick_task(label, query, task, k):
            with st.spinner(f"AI đang soạn: {label}..."): 
                out, evidence, invalid = generate_evidence_based_wrapper(task, query, k)
            
            if not out: 
                st.warning("Không nhận được nội dung từ AI.")
                return
            
            # ==============================================================
            # BỘ LỌC ĐÁNH SỐ TỐI THƯỢNG (Bắt được mọi loại dấu gạch ngang của AI)
            # ==============================================================
            citation_mapping = {}
            current_index = 1
            
            def replacer(match):
                nonlocal current_index
                # Lấy phần mã băm (ví dụ: 3FD1BE479E) bất chấp dấu gạch ngang phía trước
                hash_code = match.group(1) or match.group(2)
                core_id = f"SRC-{hash_code.upper()}"
                
                if core_id not in citation_mapping:
                    citation_mapping[core_id] = current_index
                    current_index += 1
                return f"[{citation_mapping[core_id]}]"
                
            # Regex siêu mạnh: Bắt mọi cụm có chữ SRC, không quan tâm AI chế ra dấu gạch gì
            pattern = r'(?:\[[^\[\]\n]*SRC[^a-zA-Z0-9]?([a-zA-Z0-9]+)[^\[\]\n]*\])|(?:SRC[^a-zA-Z0-9]?([a-zA-Z0-9]+))'
            clean_out = re.sub(pattern, replacer, out, flags=re.IGNORECASE)
            
            # Đồng bộ danh sách chuẩn vào bộ nhớ hệ thống (ĐỂ TAB 9 NHẬN DIỆN ĐƯỢC)
            new_refs = []
            docs = st.session_state.get("documents", {})
            for ref_id, idx in citation_mapping.items():
                new_refs.append({
                    "ref_id": ref_id,
                    "vancouver_index": idx,
                    "metadata": docs.get(ref_id, {})
                })
            
            st.session_state["current_references"] = new_refs
            st.session_state["citation_registry"] = citation_mapping
            st.session_state["last_generated"] = clean_out
            # ==============================================================

            with result_box:
                st.write("---")
                st.subheader(label)
                # In ra bản văn bản ĐÃ ĐƯỢC ÉP SỐ THỨ TỰ (clean_out)
                st.markdown(clean_out) 
                
                st.markdown("### 🔎 Dấu vết bằng chứng (Evidence Trace)")
                refs = st.session_state.get("current_references", [])
                for ref in refs:
                    vi = ref.get("vancouver_index", "")
                    rid = ref.get("ref_id", "")
                    sid = rid.replace("REF-", "") if rid.startswith("REF-") else rid
                    meta = ref.get("metadata", {}) or {}
                    
                    with st.expander(f"[{vi}] ↳ {meta.get('title','Tài liệu chưa có tiêu đề')[:85]}..."):
                        st.write(f"**Tệp gốc:** `{meta.get('file_name','N/A')}`")
                        if meta.get("doi"): 
                            st.write(f"**DOI:** {meta['doi']}")
                        for ch in [x for x in evidence if x.get("source_id") == sid]:
                            st.markdown(f"- **Trang/Mục:** `{ch.get('page','N/A')}` | **Độ khớp:** `{ch.get('score',0):.4f}` | **Mã đoạn:** `{ch.get('chunk_id','N/A')}`")
                            st.info(f"_{ch.get('text','')}_")
                            
                with st.expander("📖 Danh mục Tài liệu tham khảo (Của bản nháp này)"): 
                    st.code(citation_bibliography_wrapper() or "Chưa có citation registry.", language="text")
                
                try: 
                    audit = Audit_generated_text_wrapper(clean_out)
                except Exception as exc: 
                    audit = {"warnings": []}
                    st.warning(f"Không thể chạy Audit tự động: {exc}")
                
                x, y = st.columns(2)
                with x: 
                    if invalid:
                        st.error(f"Phát hiện citation ảo: {', '.join(invalid)}")
                    else:
                        st.success("Không phát hiện citation ảo.")
                with y: 
                    if audit.get("warnings"):
                        st.warning(f"Số liệu lạ (Cần Audit lại): {', '.join(audit.get('warnings',[]))}")
                    else:
                        st.success("Không phát hiện số liệu lạ ngoài bằng chứng.")
                
                st.session_state["Audit_log"].append({"type": label, "invalid_citation": invalid, "Audit": audit})
                st.session_state["Audit_log"] = st.session_state["Audit_log"][-100:]
            with result_box:
                st.write("---"); st.subheader(label); st.markdown(out); st.markdown("### 🔎 Dấu vết bằng chứng (Evidence Trace)")
                refs=st.session_state.get("current_references",[])
                for ref in refs:
                    vi=ref.get("vancouver_index",""); rid=ref.get("ref_id",""); sid=rid.replace("REF-","") if rid.startswith("REF-") else rid; meta=ref.get("metadata",{}) or {}
                    with st.expander(f"[{vi}] ↳ {meta.get('title','Tài liệu chưa có tiêu đề')[:85]}..."):
                        st.write(f"**Tệp gốc:** `{meta.get('file_name','N/A')}`")
                        if meta.get("doi"): st.write(f"**DOI:** {meta['doi']}")
                        for ch in [x for x in evidence if x.get("source_id")==sid]:
                            st.markdown(f"- **Trang/Mục:** `{ch.get('page','N/A')}` | **Độ khớp:** `{ch.get('score',0):.4f}` | **Mã đoạn:** `{ch.get('chunk_id','N/A')}`"); st.info(f"_{ch.get('text','')}_")
                with st.expander("📖 Danh mục Tài liệu tham khảo (Của bản nháp này)"): st.code(citation_bibliography_wrapper() or "Chưa có citation registry.",language="text")
                
                try: 
                    audit=Audit_generated_text_wrapper(out)
                except Exception as exc: 
                    audit={"warnings":[]}
                    st.warning(f"Không thể chạy Audit tự động: {exc}")
                
                x,y=st.columns(2)
                with x: 
                    if invalid:
                        st.error(f"Phát hiện citation ảo: {', '.join(invalid)}")
                    else:
                        st.success("Không phát hiện citation ảo.")
                with y: 
                    if audit.get("warnings"):
                        st.warning(f"Số liệu lạ (Cần Audit lại): {', '.join(audit.get('warnings',[]))}")
                    else:
                        st.success("Không phát hiện số liệu lạ ngoài bằng chứng.")
                
                st.session_state["Audit_log"].append({"type":label,"invalid_citation":invalid,"Audit":audit})
                st.session_state["Audit_log"]=st.session_state["Audit_log"][-100:]

        st.subheader("📝 Lệnh viết nhanh"); b1,b2,b3,b4,b5=st.columns(5)
        with b1: btn1=st.button("Đặt vấn đề",key=ui_key("btn_dat_van_de"))
        with b2: btn2=st.button("Tổng quan tài liệu",key=ui_key("btn_tong_quan"))
        with b3: btn3=st.button("Phương pháp NC",key=ui_key("btn_phuong_phap"))
        with b4: btn4=st.button("Bàn luận KQNC và So sánh",key=ui_key("btn_ban_luan"))
        with b5: btn5=st.button("Trích dẫn TLTK",key=ui_key("btn_tltk"))
        st.write("---"); st.subheader("Lệnh tùy chỉnh")
        custom_prompt=st.text_area("Nhập câu lệnh khác:",key=ui_key("custom_prompt_tab3")); k_custom=st.slider("Số nguồn bằng chứng truy xuất",3,MAX_TOP_K,DEFAULT_TOP_K,key=ui_key("top_k_tab3")); btnc=st.button("▶️ Chạy lệnh tùy chỉnh",key=ui_key("btn_custom"))
        
        if btn1: run_quick_task("Đặt vấn đề","Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, dịch tễ học, gánh nặng bệnh tật liên quan sử dụng thuốc",f"Viết phần 'Đặt vấn đề'. Viết thành MỘT MẠCH VĂN LIỀN MẠCH, khoảng 400 từ, gồm 3-4 đoạn văn.\n{citation_rules}",6)
        if btn2: run_quick_task("Tổng quan tài liệu","Tổng quan y văn, các nghiên cứu liên quan, cơ chế dược lý, kết quả chính, khuyến cáo điều trị",f"Viết phần 'Tổng quan tài liệu' chuyên sâu.\n{citation_rules}",8)
        if btn3: run_quick_task("Phương pháp nghiên cứu","Đối tượng nghiên cứu, tiêu chuẩn chọn loại, thiết kế nghiên cứu, cỡ mẫu, biến số nghiên cứu",f"Viết 'Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU'.\n{citation_rules}",5)
        if btn4:
            if not my_research_data.strip() and not my_table_remarks.strip(): 
                st.warning("⚠️ Cần dán bảng số liệu (ô 1) và nhận xét (ô 2) vào phía trên trước!")
            else:
                ctx=f"SỐ LIỆU BẢNG:\n{my_research_data}\n\nNHẬN XÉT DIỄN GIẢI:\n{my_table_remarks}"
                run_quick_task("Bàn luận và So sánh toàn diện",ctx,f"DỮ LIỆU NGHIÊN CỨU:\n{ctx}\nYÊU CẦU: Viết BÀN LUẬN TOÀN DIỆN. Giải thích nguyên nhân và so sánh với y văn. Viết liền mạch, không dùng nhãn phân chia.\n{citation_rules}",8)
        if btn5:
            query="Tài liệu tham khảo, tác giả, năm xuất bản, tạp chí"
            task=f"Chọn các SOURCE_TAG phù hợp làm tài liệu tham khảo chính và để hệ thống tạo danh mục Vancouver.\n{citation_rules}"
            run_quick_task("Trích dẫn TLTK",query,task,10)
        if btnc:
            if not custom_prompt.strip(): 
                st.warning("Vui lòng nhập yêu cầu!")
            else: 
                run_quick_task("Kết quả lệnh tùy chỉnh",custom_prompt,f"{custom_prompt}\n{citation_rules}",k_custom)
        
        st.write("---"); st.subheader("📄 Xuất Bản Nháp")
        if st.button("📥 Tải bản nháp hiện tại ra file Word",use_container_width=True,type="primary",key=ui_key("export_current_draft")):
            if not st.session_state.get("last_generated"): 
                st.warning("Chưa có bản nháp. Vui lòng chạy một lệnh viết luận văn trước.")
            else:
                db=create_word_document("Bản nháp hỗ trợ nghiên cứu",st.session_state["last_generated"],citation_bibliography_wrapper())
                st.download_button("Bấm vào đây để tải file",data=db,file_name="Ban_nhap.docx",mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True,key=ui_key("download_current_draft"))
        render_writing_chat()

    # ------------------------------------------------------------
    # TAB 6 – VIẾT BẰNG OLLAMA LOCAL 
    # ------------------------------------------------------------
    with tabs[5]:
        render_ollama_writer_tab()

    # ------------------------------------------------------------
    # TAB 7 – PHÂN TÍCH SỐ LIỆU 
    # ------------------------------------------------------------
    with tabs[6]:
        st.header("📊 Phân tích số liệu bệnh án (SPSS Mini)")
        excel_file=st.file_uploader("Tải file Excel",type=["xlsx","xls"],key=ui_key("excel_uploader_tab4"))
        if excel_file is not None:
            with st.spinner("Đang dọn dẹp và nạp dữ liệu..."):
                raw=safe_read_excel(excel_file)
            if not raw.empty:
                st.success(f"✅ Nạp thành công: {raw.shape[0]} dòng và {raw.shape[1]} cột."); st.dataframe(raw.head(50),use_container_width=True); st.session_state["excel_data"]=raw
                try:
                    with st.spinner("Đang dọn dẹp và chuẩn hóa dữ liệu bằng Data Engine..."): df,logs=auto_clean_data(raw)
                    st.session_state["excel_df"]=df; st.session_state["clean_logs"]=logs
                except Exception as exc: st.error(f"Lỗi khi chuẩn hóa dữ liệu: {exc}")
        render_chat_assistant()
        df=st.session_state.get("excel_df")
        if df is not None and not df.empty:
            for log in st.session_state.get("clean_logs",[]): pass
            if st.session_state.get("clean_logs"):
                with st.expander("🛠️ Xem nhật ký tự động dọn dẹp dữ liệu",expanded=True):
                    for log in st.session_state["clean_logs"]: st.write(log)
            for item in validate_dataframe(df) or []: st.warning(item)
            with st.expander("Xem dữ liệu sau khi chuẩn hóa"): st.dataframe(df,use_container_width=True)
            st.write("---"); st.markdown(f"### 🛒 Giỏ kết quả: **{len(st.session_state.get('result_cart',[]))}** bảng đã lưu"); st.info("💡 Mỗi khi anh bấm các nút thống kê bên dưới, kết quả tự động được nạp vào Giỏ này để lát nữa AI tuyển chọn hoặc xuất ra Word.")
            gc1,gc2=st.columns(2)
            with gc1:
                if st.button("🗑️ Xóa toàn bộ Giỏ kết quả",use_container_width=True,key=ui_key("clear_result_cart")):
                    st.session_state["result_cart"]=[]; st.session_state["saved_tables"]={}; st.session_state["selection_decisions"]=[]; st.session_state["narrative_plan"]={}; gc.collect(); st.rerun()
            with gc2:
                saved=st.session_state.get("saved_tables",{})
                if saved:
                    md=[]
                    for tid,dft in saved.items():
                        h="| "+" | ".join(map(str,dft.columns))+" |"; sep="|"+"|".join(["---"]*len(dft.columns))+"|"; rows=["| "+" | ".join(map(str,row.values))+" |" for _,row in dft.iterrows()]; md.append(f"### Kết quả Thống kê: {tid}\n\n"+"\n".join([h,sep]+rows))
                    wd=create_word_document("Phụ lục Số liệu Thống kê (Xuất từ Giỏ kết quả)","\n\n".join(md)); st.download_button("📥 Tải TẤT CẢ bảng ra file Word",data=wd,file_name="Phu_luc_So_lieu_Thong_ke.docx",mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True,key=ui_key("download_all_tables"))
                else: st.button("📥 Tải TẤT CẢ bảng ra file Word",disabled=True,use_container_width=True,key=ui_key("download_all_tables_disabled"))

            st.write("---"); st.subheader("📋 Bộ máy tuyển chọn & Sắp xếp bảng cho Chương Kết quả")
            with st.expander("🎯 Khai báo Mục tiêu nghiên cứu"):
                o1=st.text_input("Mục tiêu 1",value="ĐẶC ĐIỂM BỆNH NHÂN NGHIÊN CỨU",key=ui_key("obj_1")); o2=st.text_input("Mục tiêu 2",value="PHÂN TÍCH THỰC TRẠNG SỬ DỤNG THUỐC",key=ui_key("obj_2"))
                objectives=[StudyObjective(id="MT1",title=o1,keywords=["tuổi","tuoi","giới","gioi","bệnh","benh","đặc điểm","nhân khẩu","bmi","SoBHYT","NgaySinh"]),StudyObjective(id="MT2",title=o2,keywords=["thuốc","thuoc","phù hợp","phu hop","liều","lieu","chỉ định","chi dinh","hoạt chất","icd","TenHang"])]
            if st.button("🚀 Chạy Table Selection Engine & Lập mạch kể chuyện",type="primary",key=ui_key("run_engine")):
                if not st.session_state["result_cart"]: st.error("❌ Giỏ kết quả đang trống! Anh cần bấm các nút thống kê để nạp số liệu vào Giỏ trước.")
                else:
                    try:
                        dec=TableSelectionEngine(objectives,st.session_state["result_cart"]).run(); st.session_state["selection_decisions"]=dec; st.session_state["narrative_plan"]=NarrativePlanner.build(dec); st.success("✅ Đã hoàn thành tuyển chọn, lọc trùng và sắp xếp cấu trúc Chương Kết quả!")
                    except Exception as exc: st.error(f"❌ Table Selection Engine lỗi: {exc}")
            if st.session_state.get("selection_decisions"):
                rows=[]
                for d in st.session_state["selection_decisions"]: rows.append({"Thứ tự":d.recommended_order or "Phụ lục","Mức độ":d.priority.value,"Hình thức":d.presentation.value,"Điểm":d.total_score,"Tiêu đề bảng":d.title,"Lý do đề xuất":d.reason})
                st.write("### 📊 Bảng tổng hợp đề xuất cấu trúc Chương 3"); st.dataframe(pd.DataFrame(rows),use_container_width=True); st.write("### 📖 Mạch kể chuyện (Result Story / Narrative Plan)"); st.json(st.session_state["narrative_plan"])

            st.write("---"); st.subheader("🚀 Trình lắp ráp tự động Chương Kết quả & Bàn luận (Auto-Assembler Agent)"); st.info("💡 Hệ thống sẽ tự động kích hoạt Loop Agent: Duyệt qua từng bảng theo mạch kể chuyện, tự viết nhận xét, tự tìm bằng chứng đối chiếu và lắp ráp thành toàn bộ bản thảo 2 chương lớn.")
            if st.button("🪄 Tự động lập bản thảo toàn bộ Chương 3 & Chương 4",type="primary",key=ui_key("btn_auto_assemble")):
                if not st.session_state.get("selection_decisions") or not st.session_state.get("saved_tables"):
                    st.warning("⚠️ Bạn cần chạy 'Table Selection Engine' để thiết lập mạch kể chuyện và lưu bảng vào Giỏ trước!")
                else:
                    try:
                        with st.spinner("Agent đang tự động quét, viết nhận xét, truy xuất y văn và lắp ráp hai chương..."):
                            a3,a4=assemble_results_and_discussion_chapter(selection_decisions=st.session_state["selection_decisions"],saved_tables=st.session_state["saved_tables"],chunks=st.session_state.get("chunks",[]),embeddings=st.session_state.get("embeddings"),bm25=st.session_state.get("bm25"),citation_engine=get_citation_engine(),study_context=st.session_state.get("study_context",{}))
                            
                            # Kích hoạt bộ lọc đánh số thứ tự nối tiếp cho cả 2 chương
                            full_draft = a3 + "\n\n---SPLIT_CHAPTER---\n\n" + a4
                            clean_draft = format_numbered_citations(full_draft)
                            clean_a3, clean_a4 = clean_draft.split("\n\n---SPLIT_CHAPTER---\n\n")
                            
                        st.session_state["assembled_ch3"]=clean_a3; st.session_state["assembled_ch4"]=clean_a4; st.success("🎉 Đã lắp ráp thành công toàn bộ bản thảo hai chương!")
                    except Exception as exc: st.error(f"❌ Auto-Assembler lỗi: {exc}")
            if st.session_state.get("assembled_ch3") and st.session_state.get("assembled_ch4"):
                t3,t4=st.tabs(["📄 Chương 3: Kết quả","📄 Chương 4: Bàn luận"])
                with t3:
                    st.markdown(st.session_state["assembled_ch3"]); d3=create_word_document("Chương 3: Kết quả nghiên cứu",st.session_state["assembled_ch3"]); st.download_button("📥 Tải Chương 3 ra file Word",data=d3,file_name="Chuong_3_Ket_qua.docx",mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",key=ui_key("dl_ch3"))
                with t4:
                    st.markdown(st.session_state["assembled_ch4"]); d4=create_word_document("Chương 4: Bàn luận",st.session_state["assembled_ch4"],citation_bibliography_wrapper()); st.download_button("📥 Tải Chương 4 ra file Word (kèm TLTK)",data=d4,file_name="Chuong_4_Ban_luan.docx",mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",key=ui_key("dl_ch4"))

            st.write("---"); st.subheader("1. Thống kê mô tả (biến phân loại)"); all_cols=df.columns.tolist(); desc_vars=st.multiselect("Chọn biến phân loại",all_cols,key=ui_key("desc_vars"))
            if st.button("Tính tần số và tỷ lệ (Tự động nạp vào Giỏ)",key=ui_key("calc_desc")):
                if not desc_vars: st.warning("Vui lòng chọn ít nhất 1 biến.")
                else:
                    n=0
                    for var in desc_vars:
                        try:
                            r=descriptive_table(df,var)
                            if r is not None and not r.empty:
                                n+=1; st.markdown(f"**► Biến: {var}**"); st.dataframe(r,use_container_width=True); rid=f"DESC_{var}"; st.session_state["saved_tables"][rid]=r; upsert_candidate(rid,f"Đặc điểm phân bố của biến {var}","demographic",[var],3.5,4.0,3.0)
                        except Exception as exc: st.error(f"⚠️ Không thể phân tích [{var}]: {exc}")
                    st.success(f"✅ Đã nạp {n} bảng mô tả vào Giỏ!")

            st.write("---"); st.subheader("2. Biến định lượng — Mô tả"); numeric_candidates=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_candidates:
                num_vars=st.multiselect("Chọn biến định lượng",numeric_candidates,key=ui_key("num_vars"))
                if st.button("Tính Mean/SD và Median/IQR (Tự động nạp vào Giỏ)",key=ui_key("calc_num")):
                    if not num_vars: st.warning("Vui lòng chọn ít nhất 1 biến.")
                    else:
                        n=0
                        for var in num_vars:
                            try:
                                s=numeric_summary(df,var)
                                if s:
                                    n+=1; num_df=pd.DataFrame([{"N":s["n"],"Mean ± SD":f"{s['mean']:.2f} ± {s['sd']:.2f}","Median (IQR)":f"{s['median']:.2f} ({s['q1']:.2f} - {s['q3']:.2f})","Min-Max":f"{s['min']:.2f} - {s['max']:.2f}"}]); st.markdown(f"**► Biến: {var}**"); st.dataframe(num_df,use_container_width=True); rid=f"NUM_{var}"; st.session_state["saved_tables"][rid]=num_df; upsert_candidate(rid,f"Đặc điểm định lượng của biến {var}","baseline",[var],3.5,4.0,3.0)
                            except Exception as exc: st.error(f"⚠️ Không thể tính [{var}]: {exc}")
                        st.success(f"✅ Đã nạp {n} bảng định lượng vào Giỏ!")

            st.write("---"); st.subheader("3. Bảng chéo và kiểm định (Chi-square / Fisher / OR / CI95)"); a,b=st.columns(2)
            with a:
                all_d=st.checkbox("✅ Chọn tất cả biến phụ thuộc",key=ui_key("chk_all_deps")); dkey=ui_key("cross_deps_all") if all_d else ui_key("cross_deps_manual"); deps=st.multiselect("Các biến phụ thuộc",all_cols,default=all_cols if all_d else [],key=dkey)
            with b:
                all_i=st.checkbox("✅ Chọn tất cả biến độc lập",key=ui_key("chk_all_indeps")); ikey=ui_key("cross_indeps_all") if all_i else ui_key("cross_indeps_manual"); indeps=st.multiselect("Các biến độc lập cần đối chiếu",all_cols,default=all_cols if all_i else [],key=ikey)
            if st.button("Quét Crosstab + Kiểm định (Nạp TẤT CẢ vào Giỏ)",key=ui_key("calc_cross")):
                if not deps or not indeps: st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
                elif not large_batch_ok("Crosstab + Kiểm định",len({frozenset([d,i]) for d in deps for i in indeps if d!=i})): st.stop()
                else:
                    done=0; seen=set()
                    with st.spinner("Đang cày xới toàn bộ ma trận số liệu..."):
                        for dep in deps:
                            for indep in indeps:
                                if dep==indep or frozenset([dep,indep]) in seen: continue
                                seen.add(frozenset([dep,indep]))
                                try:
                                    r=crosstab_test(df,indep,dep); pv=r.get("p_value"); done+=1; st.markdown(f"**► Mối liên quan giữa: [{indep}] & [{dep}] — {'🟢 CÓ Ý NGHĨA' if pv is not None and pv<0.05 else '⚪ KHÔNG Ý NGHĨA'}**"); st.dataframe(r["table"],use_container_width=True); st.write(f"- **Kiểm định:** {r.get('test','')} | **p-value:** `{pv:.4g}`" if pv is not None else f"- **Kiểm định:** {r.get('test','')}");
                                    if "effect_size" in r: st.write(f"- **Chỉ số (Effect Size / OR):** `{r['effect_size']}`")
                                    rid=f"CROSS_{indep}_{dep}"; st.session_state["saved_tables"][rid]=r["table"]; upsert_candidate(rid,f"Mối liên quan giữa {indep} và {dep}","association",[indep,dep],4.5,4.5,5.0,pv)
                                except Exception as exc: st.error(f"⚠️ Lỗi phân tích chéo [{indep} & {dep}]: {exc}")
                    st.success(f"✅ Đã phân tích và nạp {done} bảng kiểm định vào Giỏ!"); gc.collect()

            st.write("---"); st.subheader("4. So sánh biến định lượng giữa 2 nhóm (T-test / Mann-Whitney)"); a,b=st.columns(2)
            with a:
                sag=st.checkbox("✅ Chọn tất cả biến nhóm",key=ui_key("chk_all_groups")); gkey=ui_key("group_vars_all") if sag else ui_key("group_vars_manual"); groups=st.multiselect("Biến nhóm (Tự lọc biến 2 mức)",all_cols,default=all_cols if sag else [],key=gkey)
            with b:
                sav=st.checkbox("✅ Chọn tất cả biến định lượng",key=ui_key("chk_all_vals")); vkey=ui_key("val_vars_all") if sav else ui_key("val_vars_manual"); vals=st.multiselect("Biến định lượng cần so sánh",numeric_candidates or all_cols,default=(numeric_candidates or all_cols) if sav else [],key=vkey)
            if st.button("Quét kiểm định so sánh (Nạp TẤT CẢ vào Giỏ)",key=ui_key("run_group_compare")):
                if not groups or not vals: st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
                elif not large_batch_ok("So sánh 2 nhóm",sum(g!=v for g in groups for v in vals)): st.stop()
                else:
                    done=0
                    with st.spinner("Đang rà soát và tính toán..."):
                        for gv in groups:
                            for vv in vals:
                                if gv==vv: continue
                                try:
                                    r=compare_two_groups(df,gv,vv); pv=r.get("p_value"); done+=1; g1,g2=r["group_names"]; st.markdown(f"**► Sự phân bố của [{vv}] giữa 2 nhóm [{gv}] — {'🟢 KHÁC BIỆT' if pv is not None and pv<0.05 else '⚪ TƯƠNG ĐỒNG'}**"); st.write(f"- **Kiểm định:** {r.get('test','')} | **p-value:** `{pv:.4g}`" if pv is not None else f"- **Kiểm định:** {r.get('test','')}");
                                    if "effect_size" in r: st.write(f"- **Chỉ số (Effect Size):** `{r['effect_size']}`")
                                    comp=pd.DataFrame({g1:[r["group1_stats"]],g2:[r["group2_stats"]]},index=["Giá trị"]); st.dataframe(comp,use_container_width=True); rid=f"COMP_{gv}_{vv}"; st.session_state["saved_tables"][rid]=comp; upsert_candidate(rid,f"Sự khác biệt của biến {vv} giữa các nhóm {gv}","association",[gv,vv],4.5,4.5,5.0,pv)
                                except Exception as exc: st.error(f"⚠️ Lỗi so sánh [{gv} & {vv}]: {exc}")
                    st.success(f"✅ Đã nạp {done} kết quả so sánh vào Giỏ!"); gc.collect()

            st.write("---"); st.subheader("5. Hồi quy logistic nhị phân (OR và 95% CI)"); outcomes_all=[c for c in all_cols if df[c].dropna().nunique()==2]; forbidden=["unnamed","ngay","ngày","ten","tên","ma","mã","sobenhan","id"]; predictors_all=[c for c in all_cols if not any(k in str(c).lower() for k in forbidden) and df[c].dropna().nunique()>1]
            a,b=st.columns([1,2])
            with a:
                sao=st.checkbox("✅ Chọn tất cả biến kết cục",key=ui_key("chk_all_outcomes")); okey=ui_key("log_outcomes_all") if sao else ui_key("log_outcomes_manual"); outcomes=st.multiselect("Biến kết cục (Nhị phân)",outcomes_all,default=outcomes_all if sao else [],key=okey)
            with b:
                sap=st.checkbox("✅ Chọn tất cả yếu tố dự báo",key=ui_key("chk_all_predictors")); pkey=ui_key("log_predictors_all") if sap else ui_key("log_predictors_manual"); predictors=st.multiselect("Yếu tố dự báo",predictors_all,default=predictors_all if sap else [],key=pkey)
            if st.button("Chạy Logistic Regression đa biến (Nạp vào Giỏ)",key=ui_key("run_logistic")):
                if not outcomes or not predictors: st.warning("Chọn ít nhất một biến ở mỗi bên.")
                elif not large_batch_ok("Logistic Regression",len(outcomes)): st.stop()
                else:
                    done=0
                    with st.spinner("Đang xây dựng mô hình hồi quy..."):
                        for out in outcomes:
                            preds=[p for p in predictors if p!=out]
                            if not preds: continue
                            try:
                                r,summary=binary_logistic_regression(df,out,preds); done+=1; st.markdown(f"**► MÔ HÌNH HỒI QUY ĐA BIẾN CHO KẾT CỤC: [{out}]**"); st.info(summary); st.dataframe(r,use_container_width=True); rid=f"LOG_{out}"; st.session_state["saved_tables"][rid]=r; upsert_candidate(rid,f"Mô hình hồi quy logistic đánh giá yếu tố liên quan đến {out}","regression",[out]+preds,5.0,5.0,5.0)
                            except Exception as exc: st.error(f"⚠️ Không thể xây dựng mô hình cho [{out}]: {exc}")
                    st.success(f"✅ Đã nạp {done} mô hình hồi quy vào Giỏ!"); gc.collect()

            st.write("---"); st.subheader("8. Diễn giải kết quả bằng AI (Nhận xét bảng chuẩn khoa học)"); st.info("💡 Anh có thể chọn một bảng riêng lẻ hoặc chọn **'🌟 Chọn TẤT CẢ các bảng trong Giỏ'** để AI tổng hợp nhận xét toàn bộ số liệu.")
            saved=st.session_state.get("saved_tables",{}); options=["-- Chỉ dùng số liệu dán tay bên dưới --","🌟 Chọn TẤT CẢ các bảng trong Giỏ"]+list(saved.keys()); choice=st.selectbox("Lựa chọn bảng hoặc nguồn dữ liệu:",options,key=ui_key("select_ai_table")); extra=st.text_area("Số liệu bổ bổ sung hoặc yêu cầu cụ thể (nếu có):",height=100,key=ui_key("interpretation_request"))
            if st.button("🤖 AI Viết Nhận Xét Bảng",type="primary",key=ui_key("ai_interpret")):
                final=""
                if choice=="🌟 Chọn TẤT CẢ các bảng trong Giỏ":
                    if not saved: st.warning("⚠️ Giỏ kết quả đang trống, chưa có bảng nào được lưu!")
                    for k,v in saved.items(): final+=f"### BẢNG: {k}\n"+v.to_markdown(index=False)+"\n\n"
                elif choice!="-- Chỉ dùng số liệu dán tay bên dưới --": final+=f"### BẢNG: {choice}\n"+saved[choice].to_markdown(index=False)+"\n\n"
                if extra.strip(): final+=f"SỐ LIỆU / YÊU CẦU BỔ SUNG:\n{extra.strip()}"
                if not final.strip(): st.warning("⚠️ Anh chưa chọn bảng nào hoặc chưa dán số liệu!")
                else:
                    prompt=f"""{BASE_SYSTEM_RULES}
Nhiệm vụ của bạn là viết phần **'Nhận xét'** cho các bảng số liệu thống kê trong luận văn Dược lâm sàng.
QUY TẮC VÀNG BẮT BUỘC:
1. CHỈ ĐƯA RA SỐ LIỆU: Chỉ diễn giải các số liệu, tần số, tỷ lệ % nổi bật có trong bảng.
2. VĂN PHONG KHOA HỌC: Câu văn logic, ngắn gọn, dễ hiểu, mạch lạc.
3. TUYỆT ĐỐI KHÔNG BÀN LUẬN: Không giải thích nguyên nhân, không suy diễn cơ chế lâm sàng, không so sánh với các nghiên cứu khác.
4. Trình bày thành các đoạn văn xuôi y khoa liền mạch, chuẩn mực.
DỮ LIỆU ĐẦU VÀO:\n{final}"""
                    try:
                        with st.spinner("AI đang phân tích số liệu và soạn nhận xét chuyên sâu..."): out=call_gemini(prompt,model=DEFAULT_MODEL)
                        if out: st.markdown("### 📝 Kết quả Nhận xét Bảng:"); st.markdown(out)
                    except Exception as exc: st.error(f"Lỗi gọi AI: {exc}")

    # ------------------------------------------------------------
    # TAB 8 – AUDIT
    # ------------------------------------------------------------
    with tabs[7]:
        st.header("🔎 Audit luận văn toàn diện")
        st.markdown('<div class="warning-box">⚠️ <b>Giới hạn cần biết:</b> Công cụ chỉ báo nguy cơ.</div>',unsafe_allow_html=True)
        text=st.text_area("Dán đoạn văn cần Audit vào đây:",height=250,key=ui_key("Audit_text")); c1,c2,c3,c4,c5,c6=st.columns(6); st.write("---"); box=st.container()
        
        with c1:
            if st.button("🔢 Số liệu",use_container_width=True,key=ui_key("Audit_numbers")):
                if not text.strip(): 
                    st.warning("Chưa có văn bản.")
                else:
                    try: 
                        r=Audit_generated_text_wrapper(text)
                    except Exception as exc: 
                        r={"exact_matches":[],"derived_matches":[],"warnings":[],"evidence_used":[]}
                        st.error(f"❌ Không thể Audit số liệu: {exc}")
                    
                    with box:
                        st.markdown("### 🔢 Kết quả Audit Số liệu")
                        st.success("**Level 1 (Khớp chính xác):** " + (", ".join(r.get("exact_matches",[])) or "Không có"))
                        st.info("**Level 2 (Khớp phái sinh):** " + (", ".join(r.get("derived_matches",[])) or "Không có"))
                        
                        if r.get("warnings"): 
                            st.error("**Level 3 (⚠️ SỐ LIỆU LẠ):** " + ", ".join(r["warnings"]))
                        else: 
                            st.success("**Level 3:** Không phát hiện số liệu lạ!")
                        
                        with st.expander("📄 Xem bằng chứng đối chiếu"):
                            for e in r.get("evidence_used",[]): st.write(f"> {e.get('text','')}")
        with c2:
            if st.button("📚 Trích dẫn",use_container_width=True,key=ui_key("Audit_citation")):
                if not text.strip(): 
                    st.warning("Chưa có văn bản.")
                else:
                    cites=re.findall(r"\[(\d+)\]",text); refs={str(x.get("vancouver_index")):x for x in st.session_state.get("current_references",[])}
                    with box:
                        st.markdown("### 📚 Kết quả Audit Citation")
                        fake=[x for x in cites if x not in refs]
                        if fake: 
                            st.error("❌ Phát hiện trích dẫn ẢO: " + ", ".join(f"[{x}]" for x in fake))
                        elif cites: 
                            st.success("✅ Toàn bộ trích dẫn khớp!")
                        else: 
                            st.info("Không tìm thấy trích dẫn [n].")
        with c3:
            if st.button("🔍 Trùng lặp",use_container_width=True,key=ui_key("Audit_overlap")):
                if not text.strip(): 
                    st.warning("Chưa có văn bản.")
                else:
                    try: ov=internal_overlap_Audit_wrapper(text)
                    except Exception as exc: ov=[]; st.error(f"❌ Không thể quét trùng lặp: {exc}")
                    with box:
                        st.markdown("### 🔍 Trùng lặp nội bộ")
                        if not ov: st.info("Không tìm thấy đoạn trùng đáng kể.")
                        for x in ov: st.markdown(f"**{x.get('file','')} – trang {x.get('page','')}**\nSimilarity: **{x.get('similarity',0)}**\n> {x.get('text','')}")
        with c4:
            if st.button("🔤 Chính tả",use_container_width=True,key=ui_key("Audit_spelling")):
                if not text.strip(): 
                    st.warning("Chưa có văn bản.")
                else:
                    p=f"{BASE_SYSTEM_RULES}\nRà soát đoạn văn bản sau để tìm lỗi chính tả/thuật ngữ. ĐOẠN VĂN: {text}"
                    try: res=call_gemini(p,model=MODEL_LITE)
                    except Exception as exc: res=f"Lỗi gọi Gemini: {exc}"
                    with box: 
                        st.markdown("### 🔤 Chính tả & Thuật ngữ\n"+str(res))
        with c5:
            if st.button("🤖 Check văn AI",use_container_width=True,key=ui_key("Audit_ai_style")):
                if not text.strip(): 
                    st.warning("Chưa có văn bản.")
                else:
                    p=f"{BASE_SYSTEM_RULES}\nSoi khắt khe các dấu hiệu văn bản do AI viết. ĐOẠN VĂN: {text}"
                    try: res=call_gemini(p,model=MODEL_LITE)
                    except Exception as exc: res=f"Lỗi gọi Gemini: {exc}"
                    with box: 
                        st.markdown("### 🤖 Chỉ báo nguy cơ AI viết\n"+str(res))
        with c6:
            if st.button("⚖️ Phản biện",use_container_width=True,key=ui_key("logic_review")):
                if not text.strip(): 
                    st.warning("Chưa có văn bản.")
                else:
                    p=f"{BASE_SYSTEM_RULES}\nĐóng vai phản biện luận văn CKI Dược lâm sàng. Chỉ ra điểm yếu logic: thiếu bằng chứng, tương quan/nhân quả, vượt giới hạn thiết kế nghiên cứu. ĐOẠN VĂN: {text}"
                    try: res=call_gemini(p)
                    except Exception as exc: res=f"Lỗi gọi Gemini: {exc}"
                    with box: 
                        st.markdown("### ⚖️ Kết quả Phản biện\n"+str(res))

    # ------------------------------------------------------------
    # TAB 9 – TLTK / METADATA
    # ------------------------------------------------------------
    with tabs[8]:
        st.header("🏷️ Tài liệu tham khảo")
        st.info("💡 Bảng dưới đây hiển thị thông tin thư mục của toàn bộ tài liệu anh đã nạp. Anh có thể nhấp đúp chuột vào từng ô để sửa thủ công.")
        docs=st.session_state.get("documents",{})
        if not docs: st.warning("⚠️ Chưa có tài liệu nào trong Evidence Database.")
        else:
            data=[]
            for sid,meta in docs.items(): data.append({"source_id":str(sid),"origin":str(meta.get("origin") or "Khác"),"authors":str(meta.get("authors") or ""),"title":str(meta.get("title") or meta.get("file_name") or ""),"journal":str(meta.get("journal") or ""),"year":str(meta.get("year") or ""),"doi":str(meta.get("doi") or "")})
            ed=st.data_editor(pd.DataFrame(data),column_config={"source_id":st.column_config.TextColumn("Mã ID",disabled=True),"origin":st.column_config.TextColumn("Nguồn",disabled=True),"authors":"Tác giả","title":"Tên bài báo / Tài liệu","journal":"Tạp chí / NXB","year":"Năm XB","doi":"DOI"},use_container_width=True,num_rows="fixed",key=ui_key("meta_editor"))
            if st.button("💾 Lưu các chỉnh sửa bảng trên",type="primary",key=ui_key("save_metadata_changes")):
                for _,row in ed.iterrows():
                    sid=str(row["source_id"])
                    if sid in st.session_state["documents"]:
                        for k in ["authors","title","journal","year","doi"]: st.session_state["documents"][sid][k]="" if pd.isna(row[k]) else str(row[k])
                st.success("✅ Đã cập nhật thông tin thư mục thành công!"); time.sleep(.8); st.rerun()
            st.write("---")
            st.subheader("🤖 AI Tự động quét và lưu Metadata hàng loạt")
            st.info("💡 Quét chunk đầu tiên của từng PDF trong kho để bổ sung Tác giả, Tên bài, Năm, Tạp chí và DOI. Có thể sửa lại bằng bảng phía trên.")
            if st.button("🚀 AI Tự động đọc toàn bộ file PDF và lưu Metadata",type="primary",key=ui_key("batch_metadata")):
                updated=0
                with st.spinner("AI đang quét metadata của toàn bộ PDF..."):
                    for sid,meta in st.session_state.get("documents",{}).items():
                        if meta.get("origin")!="PDF":
                            continue
                        target=""
                        for c in st.session_state.get("chunks",[]):
                            if _field(c,"source_id")==sid:
                                target=_field(c,"text","") or ""
                                if target: break
                        if target:
                            m=extract_metadata_from_text_ai_wrapper(target)
                            if m:
                                for k,v in m.items():
                                    if v: meta[k]=v
                                updated+=1
                st.success(f"✅ Đã cập nhật metadata cho {updated} file PDF.")
                if updated: st.rerun()

            for sid,meta in list(st.session_state.get("documents",{}).items()):
                if meta.get("origin")=="PDF":
                    with st.expander(f"🤖 AI đọc lại: {sid} — {meta.get('file_name','')}"):
                        if st.button("Đọc lại metadata file này",key=ui_key(f"ai_meta_{sid}")):
                            target=""
                            for c in st.session_state.get("chunks",[]):
                                if _field(c,"source_id")==sid:
                                    target=_field(c,"text","") or ""
                                    if target: break
                            if target:
                                m=extract_metadata_from_text_ai_wrapper(target)
                                if m:
                                    st.session_state["documents"][sid].update({k:v for k,v in m.items() if v})
                                    st.success("✅ Đã cập nhật!")
                                    st.rerun()

            st.write("---")
            st.subheader("⚙️ Danh sách domain Tạp chí Y học Việt Nam")
            domains_text=st.text_area("Mỗi domain một dòng:",value="\n".join(st.session_state.get("vn_journal_domains",DEFAULT_VN_JOURNAL_DOMAINS)),height=120,key=ui_key("domains_text"))
            if st.button("💾 Lưu danh sách domain",key=ui_key("save_domains")):
                st.session_state["vn_journal_domains"]=[d.strip() for d in domains_text.splitlines() if d.strip()]
                st.success("Đã cập nhật danh sách domain.")

            st.write("---"); st.subheader("📖 Danh sách Vancouver hiện tại")
            if st.session_state.get("citation_registry"): st.code(citation_bibliography_wrapper() or "Chưa có trích dẫn.",language="text")
            else: st.info("Chưa có trích dẫn nào được sinh ra trong bản nháp.")

if __name__ == "__main__":
    main()
