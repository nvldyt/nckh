# summarizer_engine.py
# ============================================================
# MODULE TÓM TẮT Y VĂN THUẦN TÚY BẰNG PYTHON (KHÔNG DÙNG AI)
# ============================================================

import streamlit as st

def render_summarizer_tab():
    """
    Giao diện và logic tóm tắt toàn bộ kho tài liệu từ Evidence Database.
    """
    st.subheader("⚡ 3. Tóm tắt Toàn bộ Y văn (Tự động bằng Python)")
    st.info("Module này quét toàn bộ tài liệu từ Tab 1 (PDF) và Tab 2 (Bài báo đã nạp), tự động cô đọng thành các ý chính gọn gàng mà không cần dùng đến AI hay API Key.")

    # Nút thực hiện tóm tắt
    if st.button("🚀 Tổng hợp & Tóm tắt Toàn bộ Kho Tài liệu", type="primary", key="btn_run_pure_summary"):
        docs = st.session_state.get("documents", {})
        chunks = st.session_state.get("chunks", [])

        if not docs:
            st.warning("⚠️ Kho dữ liệu đang trống! Hãy nạp tài liệu ở Tab 1 hoặc Tab 2 trước.")
            st.session_state["cached_summary"] = ""
        else:
            with st.spinner("Python đang cô đọng và tổng hợp dữ liệu..."):
                # Xây dựng cấu trúc bản tóm tắt thuần túy
                summary_text = "### TỔNG HỢP Y VĂN NGHIÊN CỨU\n\n"
                summary_text += f"- Tổng số nguồn tài liệu đã nạp: **{len(docs)}**\n"
                summary_text += f"- Tổng số đoạn phân tích (chunks): **{len(chunks)}**\n\n---\n\n"

                for sid, meta in docs.items():
                    title = meta.get("title") or meta.get("file_name") or "Tài liệu không tên"
                    origin = meta.get("origin", "Tài liệu nội bộ / PDF")
                    authors = meta.get("authors", "Chưa rõ tác giả")
                    year = meta.get("year", "N/A")

                    # Lọc các đoạn text (chunks) thuộc về tài liệu này
                    doc_chunks = [c.get("text", "") for c in chunks if c.get("source_id") == sid]
                    
                    # Ghép các đoạn đầu tiên để làm đoạn trích xuất đại diện (khoảng 1000 ký tự)
                    combined_text = " ".join(doc_chunks).strip()
                    snippet = combined_text[:4000] if combined_text else "Không có nội dung văn bản bóc tách."

                    summary_text += f"#### [{sid}] {title}\n"
                    summary_text += f"- **Nguồn:** {origin} | **Tác giả:** {authors} ({year})\n"
                    summary_text += f"- **Nội dung trích rút cốt lõi:** _{snippet}..._\n\n"
                    summary_text += "---\n\n"

                # Lưu vào session_state để các tab khác (như tab Ollama) có thể gọi dùng chung
                st.session_state["cached_summary"] = summary_text
            st.success("✅ Đã tóm tắt thành công toàn bộ kho tài liệu!")

    # Hiển thị kết quả tóm tắt nếu đã có trong bộ nhớ đệm
    cached = st.session_state.get("cached_summary", "")
    if cached:
        st.write("### 📋 Kết quả Tóm tắt Sẵn sàng:")
        st.text_area(
            "Bản tóm tắt văn bản (Dùng để chuyển sang Tab 5 cho Ollama xử lý):", 
            value=cached, 
            height=300, 
            key="txt_pure_summary_display"
        )
        
        # Nút tải file tóm tắt về máy
        st.download_button(
            label="📥 Tải bản tóm tắt (TXT)",
            data=cached,
            file_name="Tom_tat_y_van.txt",
            mime="text/plain",
            key="dl_pure_summary_file"
        )
