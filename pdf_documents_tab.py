# pdf_documents_tab.py
import streamlit as st
import pandas as pd
import gc
import time

def render_pdf_documents_tab(
    ui_key,
    render_evidence_database_status,
    add_pdf_documents,
    retrieve_evidence_wrapper,
    MAX_TOP_K,
    DEFAULT_TOP_K,
    extract_metadata_from_text_ai_wrapper,
    parse_vancouver_text_wrapper, # <--- THÊM BIẾN NÀY
    _field
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
            
    # =========================================================
    # QUẢN LÝ METADATA ĐƯỢC CHUYỂN VỀ ĐÚNG TAB 2
    # =========================================================
    st.write("---")
    st.subheader("🏷️ Quản lý & Cập nhật Thông tin thư mục (Metadata)")
    st.info("💡 Bảng dưới đây quản lý (Tác giả, Năm, Tạp chí...) để AI tự động trích dẫn chuẩn Vancouver. Anh có thể sửa tay hoặc nhờ AI quét hàng loạt.")
    
    docs = st.session_state.get("documents", {})
    if not docs: 
        st.warning("⚠️ Chưa có tài liệu nào trong Evidence Database.")
    else:
        data = []
        for sid, meta in docs.items(): 
            data.append({
                "source_id": str(sid), 
                "origin": str(meta.get("origin") or "Khác"), 
                "authors": str(meta.get("authors") or ""), 
                "title": str(meta.get("title") or meta.get("file_name") or ""), 
                "journal": str(meta.get("journal") or ""), 
                "year": str(meta.get("year") or ""), 
                "doi": str(meta.get("doi") or "")
            })
        
        ed = st.data_editor(
            pd.DataFrame(data), 
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
            key=ui_key("meta_editor_tab2")
        )
        
        c_btn1, c_btn2 = st.columns([1, 1])
        with c_btn1:
            if st.button("💾 Lưu các chỉnh sửa bảng trên", type="primary", key=ui_key("save_metadata_changes_tab2")):
                for _, row in ed.iterrows():
                    sid = str(row["source_id"])
                    if sid in st.session_state["documents"]:
                        for k in ["authors", "title", "journal", "year", "doi"]: 
                            st.session_state["documents"][sid][k] = "" if pd.isna(row[k]) else str(row[k])
                st.success("✅ Đã cập nhật thông tin thư mục thành công!")
                time.sleep(0.8)
                st.rerun()
        with c_btn2:
            if st.button("🚀 AI tự động đọc và điền Metadata cho file PDF", key=ui_key("batch_metadata_tab2")):
                updated = 0
                with st.spinner("AI đang lùng sục siêu dữ liệu (Bỏ qua quảng cáo & Quét 15 đoạn đầu)..."):
                    for sid, meta in st.session_state.get("documents", {}).items():
                        if meta.get("origin") != "PDF": continue
                        
                        # Gom 15 đoạn text đầu tiên (khoảng 3-4 trang) để AI quét
                        first_chunks = []
                        for c in st.session_state.get("chunks", []):
                            # Dùng str() để đảm bảo ID khớp nhau tuyệt đối
                            if str(_field(c, "source_id")) == str(sid):
                                text_chunk = _field(c, "text", "") or ""
                                if text_chunk: first_chunks.append(text_chunk)
                        
                        target = "\n".join(first_chunks[:15])[:15000]

                        if target:
                            m = extract_metadata_from_text_ai_wrapper(target)
                            if m:
                                for k, v in m.items():
                                    k_lower = str(k).lower().strip() # Ép key về chữ thường (title, authors...)
                                    v_str = str(v).strip()
                                    
                                    if k_lower in ["title", "authors", "journal", "year", "doi"]:
                                        # Bỏ qua nếu AI trả về giá trị rỗng hoặc chữ null
                                        if v_str and v_str.lower() not in ["null", "none", "unknown", "n/a", "na", ""]:
                                            # ĐÁNH CHẶN: Nếu AI vẫn cố chấp lấy tên file .pdf thì vứt bỏ
                                            if k_lower == "title" and v_str.lower().endswith(".pdf"):
                                                continue
                                            meta[k_lower] = v_str
                                updated += 1
                                
                st.success(f"✅ Đã cập nhật metadata cho {updated} file PDF.")
                if updated: 
                    # QUAN TRỌNG NHẤT: Bắn tín hiệu ép Streamlit XÓA CACHE và vẽ lại bảng mới
                    st.session_state["ui_version"] = st.session_state.get("ui_version", 0) + 1
                    time.sleep(0.8)
                    st.rerun()
    # --- TÍNH NĂNG MỚI: DÁN TRÍCH DẪN ---
        with st.expander("🪄 DÁN DANH SÁCH TRÍCH DẪN VÀO ĐÂY ĐỂ AI TỰ ĐỘNG KHỚP VÀO BẢNG", expanded=False):
            pasted_bib = st.text_area("Dán các dòng trích dẫn (VD: Vancouver) vào đây:", height=150, key=ui_key("pasted_bib_text"))
            if st.button("⚡ Phân tích và Tự động điền vào bảng", type="primary", key=ui_key("parse_bib_btn")):
                if not pasted_bib.strip():
                    st.warning("Vui lòng dán văn bản trước!")
                else:
                    with st.spinner("AI đang bóc tách và khớp dữ liệu với các file hiện tại..."):
                        # Tạo từ điển {ID: Tên file} để gửi cho AI làm gốc đối chiếu
                        docs_info = {sid: meta.get("title") or meta.get("file_name") or "" for sid, meta in st.session_state.get("documents", {}).items()}
                        
                        parsed_data = parse_vancouver_text_wrapper(pasted_bib, docs_info)
                        if parsed_data:
                            updated = 0
                            for sid, new_meta in parsed_data.items():
                                if sid in st.session_state["documents"]:
                                    for k in ["authors", "title", "journal", "year", "doi"]:
                                        val = new_meta.get(k, "")
                                        if val and val.lower() not in ["null", "none", "n/a", "na", "unknown", ""]:
                                            st.session_state["documents"][sid][k] = val
                                    updated += 1
                            if updated:
                                st.success(f"✅ Đã khớp và cập nhật thành công {updated} tài liệu!")
                                st.session_state["ui_version"] = st.session_state.get("ui_version", 0) + 1
                                time.sleep(0.8)
                                st.rerun()
                            else:
                                st.warning("AI đã bóc tách nhưng không khớp được với tài liệu nào trong bảng.")
                        else:
                            st.error("❌ Lỗi: AI không trích xuất được thông tin hợp lệ.")
    # =========================================================
    st.write("---")
    st.subheader("🔎 Tìm bằng chứng trong toàn bộ Evidence Database")
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
