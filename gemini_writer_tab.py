# gemini_writer_tab.py
import os
import io
import time
import json
import gc
import re
import random
import pandas as pd
import streamlit as st
from docx import Document
from google import genai

from writing_engine import call_gemini, BASE_SYSTEM_RULES, DEFAULT_MODEL, MODEL_LITE
from synthesis_engine import build_literature_matrix
from audit_engine import Audit_generated_text

def get_gemini_key() -> str:
    """Tự động xoay vòng lấy API Key từ danh sách trong st.secrets."""
    try:
        if "GEMINI" in st.secrets and "API_KEYS" in st.secrets["GEMINI"]:
            keys = st.secrets["GEMINI"]["API_KEYS"]
            if keys and isinstance(keys, list):
                return random.choice(keys).strip()
            elif isinstance(keys, str):
                return random.choice([k.strip() for k in keys.split(",") if k.strip()])
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")

def render_gemini_writer_tab(
    ui_key, 
    render_evidence_database_status, 
    build_literature_matrix, 
    create_word_document, 
    generate_evidence_based_wrapper, # Không dùng nữa nhưng giữ lại để không lỗi app.py
    get_citation_engine, 
    citation_bibliography_wrapper, 
    Audit_generated_text_wrapper, 
    internal_overlap_Audit_wrapper, 
    _field,
    format_numbered_citations,
    extract_metadata_from_text_ai_wrapper
):
    if "Audit_log" not in st.session_state:
        st.session_state["Audit_log"] = []
    if "study_context" not in st.session_state:
        st.session_state["study_context"] = {}
    if "ai_pending_remark" not in st.session_state:
        st.session_state["ai_pending_remark"] = ""
    
    st.header("📝 Viết tự động bằng AI (RAG Pro + Flash Streaming)")
    st.caption("✍️ Tốc độ nhả chữ siêu tốc. Tích hợp Audit & Trích dẫn tự động.")
    st.warning("⚠️ Đây là công cụ tạo bản nháp. Mọi trích dẫn và số liệu cần được rà soát lại trước khi in.")
    
    render_evidence_database_status("Dữ liệu này sẽ được nhồi trực tiếp vào siêu bộ nhớ của Gemini Flash.")
    
    active_key = get_gemini_key()
    if not active_key:
        st.error("❌ Hệ thống không tìm thấy Gemini API Key trong Secrets. Vui lòng cấu hình trước khi dùng.")

    # =====================================================================
    # 0. CHUẨN BỊ SIÊU BỘ NHỚ (NẠP TOÀN BỘ RAG CHO FLASH)
    # =====================================================================
    docs = st.session_state.get("documents", {})
    chunks = st.session_state.get("chunks", [])
    
    doc_mapping_id_to_num = {}
    doc_mapping_num_to_meta = {}
    context_blocks = []
    
    if docs:
        ref_counter = 1
        doc_list_text = []
        for sid, meta in docs.items():
            doc_mapping_id_to_num[sid] = ref_counter
            doc_mapping_num_to_meta[str(ref_counter)] = {"ref_id": sid, "metadata": meta}
            authors = meta.get("authors", "Không rõ tác giả")
            title = meta.get("title") or meta.get("file_name") or sid
            year = meta.get("year", "")
            doc_list_text.append(f"[{ref_counter}] {authors}. {title}. {year}")
            ref_counter += 1
            
        context_blocks.append("DANH MỤC TÀI LIỆU GỐC:\n" + "\n".join(doc_list_text))
        
        ev_lines = []
        for c in chunks:
            sid = c.get("source_id")
            doc_idx = doc_mapping_id_to_num.get(sid, "?")
            text = c.get("text", "")
            if text.strip():
                ev_lines.append(f"[Trích đoạn từ tài liệu {doc_idx}]:\n{text}")
        context_blocks.append("NỘI DUNG CHI TIẾT TỪ CÁC TÀI LIỆU:\n" + "\n---\n".join(ev_lines))
        
    compiled_context = "\n\n".join(context_blocks)

    # =====================================================================
    # 1. KHAI BÁO BỐI CẢNH NGHIÊN CỨU (STUDY CONTEXT)
    # =====================================================================
    with st.expander("🎯 KHAI BÁO BỐI CẢNH NGHIÊN CỨU (Cấu hình 1 lần - AI dùng mãi mãi)", expanded=True):
        ctx = st.session_state["study_context"]
        a, b = st.columns(2)
        with a:
            ctx_title = st.text_input("Tên đề tài:", value=ctx.get("title", ""), key=ui_key("ctx_title"))
            ctx_design = st.text_input("Thiết kế NC:", value=ctx.get("design", ""), key=ui_key("ctx_design"))
            ctx_population = st.text_input("Đối tượng:", value=ctx.get("population", ""), key=ui_key("ctx_population"))
        with b:
            ctx_sample = st.text_input("Cỡ mẫu (N=):", value=ctx.get("sample_size", ""), key=ui_key("ctx_sample"))
            ctx_obj = st.text_area("Mục tiêu chính:", value=ctx.get("objectives", ""), height=110, key=ui_key("ctx_objectives"))
            
        if st.button("💾 Lưu Cấu Hình Bối Cảnh", key=ui_key("save_study_context"), type="primary"):
            st.session_state["study_context"] = {"title": ctx_title, "design": ctx_design, "population": ctx_population, "sample_size": ctx_sample, "objectives": ctx_obj}
            st.success("✅ Đã đồng bộ hóa bối cảnh đề tài vào não AI.")

    # =====================================================================
    # 2. MA TRẬN TỔNG HỢP Y VĂN
    # =====================================================================
    with st.expander("🌟 Tự động lập Ma trận Tổng hợp Y văn từ Evidence Database", expanded=False):
        if st.button("🚀 Khởi tạo Ma trận", type="primary", key=ui_key("btn_build_matrix")):
            if not docs: 
                st.warning("⚠️ Hãy nạp tài liệu PDF/Bài báo trước.")
            else:
                with st.spinner("⏳ Đang cấu trúc hóa ma trận..."):
                    try: 
                        matrix_df = build_literature_matrix(docs, chunks)
                        if not matrix_df.empty:
                            st.session_state["literature_matrix_df"] = matrix_df
                            st.success("✅ Đã lập thành công Ma trận!")
                            st.dataframe(matrix_df, use_container_width=True)
                            mb = create_word_document("Ma trận Y văn", "### Ma trận tổng hợp y văn\n\n" + matrix_df.to_markdown(index=False))
                            st.download_button("📥 Tải Ma trận (Word)", data=mb, file_name="Ma_tran.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=ui_key("download_matrix_word"))
                    except Exception as exc: 
                        st.error(f"❌ Lỗi: {exc}")

    # =====================================================================
    # 3. DỮ LIỆU NGHIÊN CỨU CỦA BẠN (CHO PHẦN BÀN LUẬN)
    # =====================================================================
    st.markdown("### 📊 Nạp số liệu nghiên cứu của bạn")
    if st.session_state.get("ai_pending_remark"):
        st.session_state[ui_key("my_table_remarks")] = st.session_state["ai_pending_remark"]
        st.session_state["ai_pending_remark"] = ""
        
    c1, c2 = st.columns(2)
    with c1: 
        my_research_data = st.text_area("1. Số liệu bảng (Dán bảng Excel/SPSS):", height=180, key=ui_key("my_research_data"))
    with c2: 
        my_table_remarks = st.text_area("2. Nhận xét bảng (Chỉ diễn giải, KHÔNG bàn luận):", height=180, key=ui_key("my_table_remarks"))
        
    if st.button("✨ AI Viết Nhận Xét Bảng", key=ui_key("write_table_remark")):
        if not my_research_data.strip(): 
            st.warning("⚠️ Vui lòng dán bảng số liệu vào ô số 1 trước!")
        else:
            prompt = f"{BASE_SYSTEM_RULES}\nNHIỆM VỤ: Ngắn gọn, CHỈ diễn giải số liệu nổi bật. BẢNG SỐ LIỆU:\n{my_research_data}"
            try:
                client = genai.Client(api_key=active_key)
                response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
                if response.text:
                    st.session_state["ai_pending_remark"] = response.text
                    st.rerun()
            except Exception as exc: 
                st.error(f"❌ Lỗi gọi AI: {exc}")

    # =====================================================================
    # 4. HỆ THỐNG XỬ LÝ LỆNH VIẾT (CÓ STREAMING SIÊU TỐC)
    # =====================================================================
    result_box = st.container()

    def run_quick_task(label, task):
        if not active_key: return
        client = genai.Client(api_key=active_key)
        
        ctx = st.session_state.get("study_context", {})
        study_context_str = f"Tên đề tài: {ctx.get('title', '')}\nThiết kế: {ctx.get('design', '')}\nĐối tượng: {ctx.get('population', '')}\nCỡ mẫu: {ctx.get('sample_size', '')}\nMục tiêu: {ctx.get('objectives', '')}"
        
        system_instruction = (
            "Bạn là một chuyên gia Dược lâm sàng xuất sắc, hỗ trợ viết luận văn Chuyên khoa I.\n"
            f"🎯 BỐI CẢNH ĐỀ TÀI CỦA NGHIÊN CỨU VIÊN:\n{study_context_str}\n\n"
            "YÊU CẦU LẬP LUẬN BẮT BUỘC:\n"
            "1. Bám sát Bối cảnh đề tài. Phân tích và tổng hợp điểm tương đồng/khác biệt giữa các tài liệu.\n"
            "2. TRÍCH DẪN: Bắt buộc dùng số thứ tự tài liệu trong ngoặc vuông, ví dụ: [1], [2]. Trích dẫn gộp dạng [2, 5].\n"
            "3. TÍNH CHUẨN XÁC: Thông tin phải khớp 100% với tài liệu cung cấp. KHÔNG bịa số liệu.\n"
            "4. VĂN PHONG: Khách quan, khoa học.\n\n"
            f"=== DỮ LIỆU Y VĂN ĐÃ NẠP (RAG) ===\n{compiled_context}"
        )
        
        with result_box:
            st.write("---")
            st.subheader(f"📝 {label}")
            
            try:
                # KÍCH HOẠT NHẢ CHỮ LIÊN TỤC (STREAMING)
                response_stream = client.models.generate_content_stream(
                    model="gemini-3.8-flash",
                    contents=task,
                    config={"system_instruction": system_instruction, "temperature": 0.2}
                )
                def stream_generator():
                    for chunk in response_stream:
                        if chunk.text: yield chunk.text
                
                clean_out = st.write_stream(stream_generator())
            except Exception as exc:
                st.error(f"❌ Lỗi Stream AI: {exc}")
                return
                
        # --- HẬU XỬ LÝ: TRÍCH XUẤT TRÍCH DẪN & KIỂM ĐỊNH SỐ LIỆU ---
        st.session_state["last_generated"] = clean_out
        
        # Bắt các số [1], [2] mà AI vừa viết ra màn hình
        found_refs = set(re.findall(r'\[(\d+)\]', clean_out))
        new_refs = []
        for ref_num in found_refs:
            if ref_num in doc_mapping_num_to_meta:
                new_refs.append({
                    "ref_id": doc_mapping_num_to_meta[ref_num]["ref_id"],
                    "vancouver_index": ref_num,
                    "metadata": doc_mapping_num_to_meta[ref_num]["metadata"]
                })
        
        new_refs = sorted(new_refs, key=lambda x: int(x["vancouver_index"]))
        st.session_state["current_references"] = new_refs
        
        with result_box:
            st.markdown("### 🔎 Dấu vết bằng chứng (Evidence Trace)")
            if not new_refs: st.info("ℹ️ Không có trích dẫn nào được sử dụng trong đoạn này.")
            for ref in new_refs:
                vi, meta = ref["vancouver_index"], ref["metadata"]
                with st.expander(f"[{vi}] ↳ {meta.get('title','Tài liệu chưa có tiêu đề')[:85]}..."):
                    st.write(f"**Tệp gốc:** `{meta.get('file_name','N/A')}`")
                    st.success("Tài liệu này đã được AI tổng hợp và đối chiếu thành công.")
            
            # Khởi chạy cỗ máy Kiểm định Toán học (Audit_engine)
            try: audit = Audit_generated_text_wrapper(clean_out)
            except Exception as exc: audit = {"warnings": []}
            
            x, y = st.columns(2)
            with x:
                max_doc = len(doc_mapping_num_to_meta)
                invalid = [num for num in found_refs if int(num) > max_doc or int(num) < 1]
                if invalid: st.error(f"🚨 Trích dẫn ảo (Hallucination): [{', '.join(invalid)}] (Vượt quá số lượng tài liệu đã nạp).")
                else: st.success("✅ Toàn bộ trích dẫn hợp lệ với y văn gốc.")
            with y:
                if audit.get("warnings"): st.warning(f"⚠️ Phát hiện số liệu không rõ nguồn gốc: {', '.join(audit.get('warnings',[]))}")
                else: st.success("✅ Không phát hiện số liệu bịa đặt.")
            
            st.session_state["Audit_log"].append({"type": label, "invalid_citation": invalid, "Audit": audit})
            st.session_state["Audit_log"] = st.session_state["Audit_log"][-50:]

    # =====================================================================
    # 5. KHU VỰC BẢNG ĐIỀU KHIỂN (NÚT BẤM)
    # =====================================================================
    st.subheader("⚡ Lệnh Viết Nhanh Luận Văn")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: btn1 = st.button("1. Đặt vấn đề", key=ui_key("btn_dat_van_de"), use_container_width=True)
    with b2: btn2 = st.button("2. Tổng quan tài liệu", key=ui_key("btn_tong_quan"), use_container_width=True)
    with b3: btn3 = st.button("3. Phương pháp NC", key=ui_key("btn_phuong_phap"), use_container_width=True)
    with b4: btn4 = st.button("4. Bàn luận & So sánh", key=ui_key("btn_ban_luan"), use_container_width=True, type="primary")
    with b5: btn5 = st.button("5. Xem trích dẫn", key=ui_key("btn_tltk"), use_container_width=True)
    
    st.write("---")
    custom_prompt = st.text_area("Yêu cầu AI Tùy chỉnh (VD: So sánh tỷ lệ kháng kháng sinh...):", key=ui_key("custom_prompt_tab3"))
    btnc = st.button("▶️ Chạy Lệnh Tùy Chỉnh", key=ui_key("btn_custom"))
    
    if btn1: run_quick_task("Đặt vấn đề", "Viết phần 'Đặt vấn đề'. Viết MỘT MẠCH VĂN LIỀN MẠCH, khoảng 500 từ, gồm 3-4 đoạn văn. Yêu cầu chèn trích dẫn.")
    if btn2: run_quick_task("Tổng quan tài liệu", "Viết phần 'Tổng quan tài liệu' chuyên sâu dựa trên y văn đã nạp. Chèn trích dẫn đầy đủ.")
    if btn3: run_quick_task("Phương pháp nghiên cứu", "Viết 'Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU' dựa vào bối cảnh tôi đã cung cấp.")
    if btn4:
        if not my_research_data.strip(): 
            st.warning("⚠️ Cần dán bảng số liệu (ô 1) vào phía trên trước khi yêu cầu AI bàn luận!")
        else:
            ctx = f"SỐ LIỆU BẢNG CỦA TÔI:\n{my_research_data}\n\nNHẬN XÉT DIỄN GIẢI:\n{my_table_remarks}"
            run_quick_task("Bàn luận và So sánh", f"DỮ LIỆU NGHIÊN CỨU CỦA TÔI:\n{ctx}\nYÊU CẦU: Viết BÀN LUẬN. Dựa vào y văn, hãy giải thích nguyên nhân của số liệu trên và so sánh đối chiếu.")
            
    if btn5:
        with result_box:
            st.write("---")
            st.subheader("📚 Quản lý Trích dẫn (Chuẩn Vancouver)")
            used_bib = citation_bibliography_wrapper()
            if used_bib:
                st.markdown("#### 📌 Các tài liệu ĐÃ TRÍCH DẪN trong bản nháp vừa tạo:")
                st.markdown(used_bib.replace("\n", "\n\n"))
            else:
                st.info("💡 Chưa có tài liệu nào được trích dẫn.")

    if btnc:
        if not custom_prompt.strip(): st.warning("⚠️ Vui lòng nhập yêu cầu!")
        else: run_quick_task("Kết quả lệnh tùy chỉnh", custom_prompt)
    
    st.write("---")
    st.subheader("📄 Xuất Bản Nháp")
    if st.button("📥 Tải bản nháp hiện tại ra file Word", use_container_width=True, type="primary", key=ui_key("export_current_draft")):
        if not st.session_state.get("last_generated"): 
            st.warning("⚠️ Chưa có bản nháp nào được tạo.")
        else:
            db = create_word_document("Bản nháp", st.session_state["last_generated"], citation_bibliography_wrapper())
            st.download_button("✅ Tải file Word (.docx)", data=db, file_name="Ban_Nhap_Luan_Van.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, key=ui_key("download_current_draft"))
