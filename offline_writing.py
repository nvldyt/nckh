import streamlit as st
import time

def render_offline_tab():
    st.title("🛠️ Tổng hợp Bản thảo & Trích xuất Nguyên văn")
    st.info("Tính năng này tìm kiếm và bóc tách nguyên văn các phần nội dung (Đặt vấn đề, Tổng quan...) từ tài liệu gốc trong Evidence Database mà không dùng AI.")

    # 1. Khu vực nhập yêu cầu
    col1, col2 = st.columns(2)
    with col1:
        section_title = st.selectbox(
            "Chọn phần cần tổng hợp nguyên văn:",
            ["Đặt vấn đề / Mở đầu", "Tổng quan tài liệu", "Phương pháp nghiên cứu", "Kết quả / Bàn luận khác"],
            key="off_title_box"
        )
    with col2:
        search_query = st.text_input("Nhập từ khóa chính xác (VD: viêm phổi cộng đồng, vancomycin):", key="off_query")

    if st.button("🚀 Trích xuất Nguyên văn", type="primary"):
        if not search_query.strip():
            st.warning("Vui lòng nhập từ khóa tìm kiếm!")
            return

        with st.spinner("Đang rà soát và gom nhóm văn bản nguyên văn từ cơ sở dữ liệu..."):
            time.sleep(0.5)
            
            try:
                from retrieval_engine import retrieve_evidence
                # Lấy số lượng lớn chunks để gom đủ nội dung
                evidence = retrieve_evidence(
                    query=search_query,
                    chunks=st.session_state.get("chunks", []),
                    matrix=st.session_state.get("embeddings"),
                    bm25=st.session_state.get("bm25"),
                    top_k=15 # Tăng số lượng để gom đủ ý
                )
            except Exception as exc:
                st.error(f"Lỗi khi tìm kiếm: {exc}")
                evidence = []

            if not evidence:
                st.warning(f"Không tìm thấy tài liệu nào khớp với từ khóa: '{search_query}'. Anh hãy kiểm tra lại kho tài liệu (Tab 2) nhé!")
                return

            # Gom nhóm các chunks theo từng nguồn file_id để tạo thành một bài/đoạn dài nguyên bản
            grouped_by_source = {}
            for ev in evidence:
                sid = ev.get("source_id", "unknown")
                if sid not in grouped_by_source:
                    grouped_by_source[sid] = {
                        "file_name": ev.get("metadata", {}).get("file_name", "Tài liệu y khoa"),
                        "chunks": []
                    }
                grouped_by_source[sid]["chunks"].append(ev.get("text", ""))

            # 2. Xây dựng nội dung trình bày nguyên văn
            output_markdown = f"### Bản trích xuất nguyên văn: {section_title}\n"
            output_markdown += f"*Từ khóa tra cứu:* `{search_query}`\n\n---\n"
            
            references = []
            ref_counter = 1

            for sid, data in grouped_by_source.items():
                ref_tag = f"[REF-{ref_counter:03d}]"
                references.append(f"- **{ref_tag}**: {data['file_name']}")
                
                output_markdown += f"#### Nguồn tài liệu: {data['file_name']} {ref_tag}\n"
                output_markdown += "> " + "\n> ".join(data["chunks"]) + "\n\n"
                output_markdown += "---\n"
                ref_counter += 1

            # 3. Hiển thị kết quả trực tiếp lên giao diện
            st.success(f"Đã trích xuất thành công nội dung nguyên văn từ {len(grouped_by_source)} nguồn tài liệu!")
            
            st.markdown(output_markdown)
            
            st.markdown("### 📖 Danh mục Tài liệu tham khảo đi kèm")
            st.markdown("\n".join(references))
            
            # Nút tải file Word hoặc TXT
            full_content = output_markdown + "\n\n### Danh mục Tài liệu tham khảo\n" + "\n".join(references)
            st.download_button(
                label="📥 Tải Bản trích xuất (TXT)",
                data=full_content,
                file_name=f"Trich_xuat_Nguyen_van_{search_query}.txt",
                mime="text/plain"
            )
