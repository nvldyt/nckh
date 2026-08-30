# gemini_writer_tab.py
import io
import time
import json
import gc
import re
import pandas as pd
import streamlit as st
from docx import Document

from writing_engine import call_gemini, BASE_SYSTEM_RULES, DEFAULT_MODEL, MODEL_LITE
from synthesis_engine import build_literature_matrix
from audit_engine import Audit_generated_text

def render_gemini_writer_tab(
    ui_key, 
    render_evidence_database_status, 
    build_literature_matrix, 
    create_word_document, 
    generate_evidence_based_wrapper, 
    get_citation_engine, 
    citation_bibliography_wrapper, 
    Audit_generated_text_wrapper, 
    internal_overlap_Audit_wrapper, 
    _field,
    format_numbered_citations,
    extract_metadata_from_text_ai_wrapper
):
    
    st.header("📝 Viết tự động bằng AI (RAG)")
    st.header("✍️ Viết luận văn dựa trên bằng chứng")
    st.warning("Đây là công cụ tạo bản nháp. Mọi citation và số liệu phải được Audit lại (đối chiếu bản gốc) trước khi đưa vào luận văn chính thức.")
    render_evidence_database_status("dùng cho các nút viết nhanh bên dưới")
    
    # =====================================================================
    # 1. KHAI BÁO BỐI CẢNH NGHIÊN CỨU (STUDY CONTEXT)
    # =====================================================================
    with st.expander("🎯 KHAI BÁO BỐI CẢNH NGHIÊN CỨU (STUDY CONTEXT) - Cấu hình 1 lần dùng mãi mãi", expanded=True):
        st.info("💡 Khai báo thông tin đề tài tại đây để AI tự động hiểu và bám sát vào mục tiêu của anh trong mọi lần sinh văn bản, không sợ lạc đề.")
        ctx = st.session_state.get("study_context", {})
        a, b = st.columns(2)
        with a:
            ctx_title = st.text_input("Tên đề tài:", value=ctx.get("title", ""), placeholder="VD: Phân tích tình hình sử dụng thuốc...", key=ui_key("ctx_title"))
            ctx_design = st.text_input("Thiết kế nghiên cứu:", value=ctx.get("design", ""), placeholder="VD: Mô tả cắt ngang hồi cứu", key=ui_key("ctx_design"))
            ctx_population = st.text_input("Đối tượng bệnh nhân:", value=ctx.get("population", ""), placeholder="VD: Bệnh nhân tăng huyết áp ngoại trú", key=ui_key("ctx_population"))
        with b:
            ctx_sample = st.text_input("Cỡ mẫu dự kiến (N=):", value=ctx.get("sample_size", ""), placeholder="VD: 150 bệnh án", key=ui_key("ctx_sample"))
            ctx_obj = st.text_area("Mục tiêu chính:", value=ctx.get("objectives", ""), height=110, placeholder="VD: 1. Khảo sát đặc điểm... 2. Đánh giá tính hợp lý...", key=ui_key("ctx_objectives"))
        if st.button("💾 Lưu Study Context", key=ui_key("save_study_context")):
            st.session_state["study_context"] = {"title": ctx_title, "design": ctx_design, "population": ctx_population, "sample_size": ctx_sample, "objectives": ctx_obj}
            st.success("✅ Đã lưu bối cảnh! Bộ não AI đã được đồng bộ hóa với đề tài của anh.")

    # =====================================================================
    # 2. MA TRẬN TỔNG HỢP Y VĂN
    # =====================================================================
    with st.expander("🌟 Tự động lập Ma trận tổng hợp y văn từ Evidence Database", expanded=False):
        st.info("💡 Tính năng này tự động quét tất cả các tài liệu / bài báo bạn đã nạp, tổng hợp thành bảng so sánh chuẩn y khoa (Tác giả, Năm, Thiết kế, Cỡ mẫu, Kết quả chính).")
        if st.button("🚀 Khởi tạo Ma trận Tổng hợp Y văn", type="primary", key=ui_key("btn_build_matrix")):
            docs = st.session_state.get("documents", {})
            chunks = st.session_state.get("chunks", [])
            if not docs: 
                st.warning("⚠️ Evidence Database đang trống! Hãy nạp tài liệu PDF hoặc bài báo từ PubMed/Tạp chí VN trước.")
            else:
                with st.spinner("AI đang phân tích và cấu trúc hóa ma trận y văn..."):
                    try: matrix_df = build_literature_matrix(docs, chunks)
                    except Exception as exc: matrix_df = pd.DataFrame(); st.error(f"❌ Lỗi khi lập ma trận: {exc}")
                if not matrix_df.empty:
                    st.session_state["literature_matrix_df"] = matrix_df
                    st.success("✅ Đã lập thành công Ma trận tổng hợp y văn!")
                    st.dataframe(matrix_df, use_container_width=True)
                    mb = create_word_document("Ma trận Tổng hợp Y văn", "### Ma trận tổng hợp y văn\n\n" + matrix_df.to_markdown(index=False))
                    st.download_button("📥 Tải Ma trận Y văn ra file Word", data=mb, file_name="Ma_tran_Tong_hop_Y_van.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=ui_key("download_matrix_word"))
                else: 
                    st.error("❌ Không thể trích xuất dữ liệu ma trận. Vui lòng thử lại.")

    # =====================================================================
    # 3. DỮ LIỆU NGHIÊN CỨU & CÁC NÚT VIẾT NHANH
    # =====================================================================
    st.markdown("### 📊 Dữ liệu nghiên cứu của riêng anh")
    if st.session_state.get("ai_pending_remark"):
        st.session_state[ui_key("my_table_remarks")] = st.session_state["ai_pending_remark"]
        st.session_state["ai_pending_remark"] = ""
    c1, c2 = st.columns(2)
    with c1: my_research_data = st.text_area("1. Số liệu bảng (Dán bảng Excel/Markdown vào đây):", height=180, key=ui_key("my_research_data"))
    with c2: my_table_remarks = st.text_area("2. Nhận xét bảng (Chỉ diễn giải số liệu, KHÔNG bàn luận):", height=180, key=ui_key("my_table_remarks"))
    if st.button("✍️ AI Viết Nhận Xét Bảng (Tự động điền vào ô 2)", key=ui_key("write_table_remark")):
        if not my_research_data.strip(): 
            st.warning("⚠️ Anh cần dán bảng số liệu vào ô số 1 trước để AI có dữ liệu đọc!")
        else:
            prompt = f"{BASE_SYSTEM_RULES}\nNHIỆM VỤ:\nNgắn gọn, CHỈ diễn giải các số liệu nổi bật. KHÔNG bàn luận, KHÔNG so sánh. Viết thành MỘT ĐOẠN VĂN LIỀN MẠCH duy nhất.\nBẢNG SỐ LIỆU:\n{my_research_data}"
            try:
                with st.spinner("AI đang phân tích bảng và soạn nhận xét..."): generated_remark = call_gemini(prompt, model=DEFAULT_MODEL)
            except Exception as exc: generated_remark = None; st.error(f"❌ Lỗi gọi Gemini: {exc}")
            if generated_remark: st.session_state["ai_pending_remark"] = generated_remark; st.rerun()

    citation_rules = "Chỉ dùng SOURCE_TAG thật để hệ thống tự chuyển thành [n]. Các số trích dẫn theo thứ tự xuất hiện."
    result_box = st.container()

    def run_quick_task(label, query, task, k):
        with st.spinner(f"AI đang soạn: {label}..."): 
            out, evidence, invalid = generate_evidence_based_wrapper(task, query, k)
        
        if not out: 
            st.warning("Không nhận được nội dung từ AI.")
            return
        
        citation_mapping = {}
        current_index = 1
        
        def replacer(match):
            nonlocal current_index
            hash_code = match.group(1) or match.group(2)
            core_id = f"SRC-{hash_code.upper()}"
            if core_id not in citation_mapping:
                citation_mapping[core_id] = current_index
                current_index += 1
            return f"[{citation_mapping[core_id]}]"
            
        pattern = r'(?:\[[^\[\]\n]*SRC[^a-zA-Z0-9]?([a-zA-Z0-9]+)[^\[\]\n]*\])|(?:SRC[^a-zA-Z0-9]?([a-zA-Z0-9]+))'
        clean_out = re.sub(pattern, replacer, out, flags=re.IGNORECASE)
        
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

        with result_box:
            st.write("---")
            st.subheader(label)
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
                    if meta.get("doi"): st.write(f"**DOI:** {meta['doi']}")
                    for ch in [x for x in evidence if x.get("source_id") == sid]:
                        st.markdown(f"- **Trang/Mục:** `{ch.get('page','N/A')}` | **Độ khớp:** `{ch.get('score',0):.4f}` | **Mã đoạn:** `{ch.get('chunk_id','N/A')}`")
                        st.info(f"_{ch.get('text','')}_")
                        
            with st.expander("📖 Danh mục Tài liệu tham khảo (Của bản nháp này)"): 
                st.code(citation_bibliography_wrapper() or "Chưa có citation registry.", language="text")
            
            try: audit = Audit_generated_text_wrapper(clean_out)
            except Exception as exc: audit = {"warnings": []}; st.warning(f"Không thể chạy Audit tự động: {exc}")
            
            x, y = st.columns(2)
            with x: 
                if invalid: st.error(f"Phát hiện citation ảo: {', '.join(invalid)}")
                else: st.success("Không phát hiện citation ảo.")
            with y: 
                if audit.get("warnings"): st.warning(f"Số liệu lạ (Cần Audit lại): {', '.join(audit.get('warnings',[]))}")
                else: st.success("Không phát hiện số liệu lạ ngoài bằng chứng.")
            
            st.session_state["Audit_log"].append({"type": label, "invalid_citation": invalid, "Audit": audit})
            st.session_state["Audit_log"] = st.session_state["Audit_log"][-100:]

    st.subheader("📝 Lệnh viết nhanh")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: btn1 = st.button("Đặt vấn đề", key=ui_key("btn_dat_van_de"))
    with b2: btn2 = st.button("Tổng quan tài liệu", key=ui_key("btn_tong_quan"))
    with b3: btn3 = st.button("Phương pháp NC", key=ui_key("btn_phuong_phap"))
    with b4: btn4 = st.button("Bàn luận KQNC và So sánh", key=ui_key("btn_ban_luan"))
    with b5: btn5 = st.button("Trích dẫn TLTK", key=ui_key("btn_tltk"))
    
    st.write("---")
    st.subheader("Lệnh tùy chỉnh")
    custom_prompt = st.text_area("Nhập câu lệnh khác:", key=ui_key("custom_prompt_tab3"))
    k_custom = st.slider("Số nguồn bằng chứng truy xuất", 3, 20, 8, key=ui_key("top_k_tab3"))
    btnc = st.button("▶️ Chạy lệnh tùy chỉnh", key=ui_key("btn_custom"))
    
    if btn1: run_quick_task("Đặt vấn đề", "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, dịch tễ học, gánh nặng bệnh tật liên quan sử dụng thuốc", f"Viết phần 'Đặt vấn đề'. Viết thành MỘT MẠCH VĂN LIỀN MẠCH, khoảng 500 từ, gồm 3-4 đoạn văn.\n{citation_rules}", 6)
    if btn2: run_quick_task("Tổng quan tài liệu", "Tổng quan y văn, các nghiên cứu liên quan, cơ chế dược lý, kết quả chính, khuyến cáo điều trị", f"Viết phần 'Tổng quan tài liệu' chuyên sâu.\n{citation_rules}", 8)
    if btn3: run_quick_task("Phương pháp nghiên cứu", "Đối tượng nghiên cứu, tiêu chuẩn chọn loại, thiết kế nghiên cứu, cỡ mẫu, biến số nghiên cứu", f"Viết 'Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU'.\n{citation_rules}", 5)
    if btn4:
        if not my_research_data.strip() and not my_table_remarks.strip(): 
            st.warning("⚠️ Cần dán bảng số liệu (ô 1) và nhận xét (ô 2) vào phía trên trước!")
        else:
            ctx = f"SỐ LIỆU BẢNG:\n{my_research_data}\n\nNHẬN XÉT DIỄN GIẢI:\n{my_table_remarks}"
            run_quick_task("Bàn luận và So sánh toàn diện", ctx, f"DỮ LIỆU NGHIÊN CỨU:\n{ctx}\nYÊU CẦU: Viết BÀN LUẬN TOÀN DIỆN. Giải thích nguyên nhân và so sánh với y văn. Viết liền mạch, không dùng nhãn phân chia.\n{citation_rules}", 8)
    if btn5:
        with result_box:
            st.write("---")
            st.subheader("📚 Trích dẫn Tài liệu tham khảo (Chuẩn Vancouver)")
            
            # 1. DANH SÁCH ĐÃ TRÍCH DẪN (Chính xác theo [1], [2] trong bài AI vừa viết)
            used_bib = citation_bibliography_wrapper()
            if used_bib:
                st.markdown("#### 📌 Các tài liệu ĐÃ TRÍCH DẪN trong bản nháp vừa tạo:")
                st.markdown(used_bib.replace("\n", "\n\n"))
            else:
                st.info("💡 Chưa có tài liệu nào được trích dẫn. Hãy chạy các lệnh viết văn (Đặt vấn đề, Tổng quan...) trước để hệ thống ghi nhận.")
            
            st.markdown("---")
            
            # 2. TOÀN BỘ KHO TÀI LIỆU
            st.markdown("#### 📂 Toàn bộ tài liệu hiện có trong Evidence Database:")
            docs = st.session_state.get("documents", {})
            if docs:
                bib_lines = []
                for i, (sid, meta) in enumerate(docs.items(), 1):
                    authors = meta.get("authors") or "Tác giả chưa xác định"
                    title = meta.get("title") or meta.get("file_name") or "Tài liệu chưa xác định"
                    journal = meta.get("journal") or "Tạp chí chưa rõ"
                    year = meta.get("year") or "Năm chưa rõ"
                    text = f"**[{i}]** {authors}. *{title}*. {journal}. {year}."
                    if meta.get("doi"): 
                        text += f" DOI: {meta['doi']}."
                    bib_lines.append(text)
                st.markdown("\n\n".join(bib_lines))
            else:
                st.warning("⚠️ Chưa có tài liệu nào trong hệ thống.")
    if btnc:
        if not custom_prompt.strip(): st.warning("Vui lòng nhập yêu cầu!")
        else: run_quick_task("Kết quả lệnh tùy chỉnh", custom_prompt, f"{custom_prompt}\n{citation_rules}", k_custom)
    
    st.write("---")
    st.subheader("📄 Xuất Bản Nháp")
    if st.button("📥 Tải bản nháp hiện tại ra file Word", use_container_width=True, type="primary", key=ui_key("export_current_draft")):
        if not st.session_state.get("last_generated"): 
            st.warning("Chưa có bản nháp. Vui lòng chạy một lệnh viết luận văn trước.")
        else:
            db = create_word_document("Bản nháp hỗ trợ nghiên cứu", st.session_state["last_generated"], citation_bibliography_wrapper())
            st.download_button("Bấm vào đây để tải file", data=db, file_name="Ban_nhap.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, key=ui_key("download_current_draft"))
    # render_writing_chat()
