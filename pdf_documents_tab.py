# pdf_documents_tab.py
import streamlit as st
import pandas as pd
import gc

def render_pdf_documents_tab(
    ui_key,
    render_evidence_database_status,
    add_pdf_documents,
    retrieve_evidence_wrapper,
    MAX_TOP_K,
    DEFAULT_TOP_K
):
    st.header("📚 Ngân hàng tài liệu gốc (PDF)")
    render_evidence_database_status()
    
    uploaded_files = st.file_uploader("Tải PDF nghiên cứu / guideline / bài báo", type=["pdf"], accept_multiple_files=True, key=ui_key("pdf_uploader"))
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 Nạp tài liệu vào Evidence Database", type="primary"):
            if not uploaded_files: 
                st.warning("Chưa có file PDF.")
            else:
                with st.spinner("Đang đọc PDF, tạo source registry và embedding..."):
                    ns, nc, errors = add_pdf_documents(uploaded_files)
                st.success(f"Đã thêm {ns} tài liệu, {nc} phân đoạn bằng chứng.")
                for e in errors: 
                    st.error(e)
    with c2:
        if st.button("🗑️ Xóa toàn bộ ngân hàng tài liệu"):
            for k, v in {"documents": {}, "chunks": [], "embeddings": None, "bm25": None, "citation_registry": {}, "last_evidence": []}.items(): 
                st.session_state[k] = v
            gc.collect()
            st.success("Đã xóa dữ liệu trong phiên hiện tại.")
            st.rerun()
            
    st.write("---")
    st.subheader("Nguồn PDF đã nạp")
    docs = list(st.session_state.get("documents", {}).values())
    if docs: 
        st.dataframe(pd.DataFrame(docs), use_container_width=True)
    else: 
        st.info("Chưa có tài liệu.")
        
    st.subheader("Tìm bằng chứng trong toàn bộ Evidence Database")
    evidence_query = st.text_area("Nhập vấn đề cần tìm trong tài liệu:", placeholder="Ví dụ: tỷ lệ bệnh nhân đạt huyết áp mục tiêu...", key=ui_key("evidence_query"))
    top_k = st.slider("Số đoạn bằng chứng", 3, MAX_TOP_K, DEFAULT_TOP_K, key=ui_key("top_k_tab1"))
    
    if st.button("🔎 Truy xuất bằng chứng", key=ui_key("retrieve_tab1")):
        if not evidence_query.strip(): 
            st.warning("Nhập câu hỏi trước.")
        else:
            evidence = retrieve_evidence_wrapper(evidence_query, k=top_k)
            st.session_state["last_evidence"] = evidence
            if not evidence: 
                st.warning("Không tìm thấy bằng chứng trong tài liệu.")
            else:
                for ev in evidence:
                    meta = st.session_state["documents"].get(ev.get("source_id"), {})
                    st.markdown(f"**{ev.get('chunk_id','')}** _({meta.get('origin','')})_\nNguồn: {ev.get('file_name','')} — Trang/mục: {ev.get('page','')}\nĐiểm: {ev.get('score',0):.4f}\n\n> {ev.get('text','')}")
