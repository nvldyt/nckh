import streamlit as st
import time

def render_offline_tab():
    st.title("🛠️ Tổng hợp Bản thảo (Không dùng AI)")
    st.info("Tính năng này bóc tách nguyên văn các bằng chứng từ Ngân hàng tài liệu (Tab 1 & 2) và tự động lắp ráp thành đoạn văn bản thô kèm chuẩn trích dẫn Vancouver. Hoạt động 100% Offline.")

    # 1. Khu vực nhập yêu cầu
    col1, col2 = st.columns(2)
    with col1:
        section_title = st.text_input("Tên phần cần viết (VD: Đặt vấn đề, Tổng quan...):", key="off_title")
    with col2:
        search_query = st.text_input("Từ khóa tìm kiếm bằng chứng (VD: tỷ lệ tử vong, kháng sinh):", key="off_query")

    if st.button("🚀 Tiến hành Tổng hợp thô", type="primary"):
        if not section_title or not search_query:
            st.warning("Vui lòng nhập đầy đủ Tên phần và Từ khóa!")
            return

        with st.spinner("Đang quét Ngân hàng tài liệu và lắp ráp bản thảo..."):
            time.sleep(1) # Tạo hiệu ứng loading cho mượt
            
            # GIẢ LẬP TÌM KIẾM (Tích hợp hàm tìm kiếm thực tế của anh vào đây)
            # Ví dụ: evidence = retrieve_evidence(search_query, k=5)
            # Ở đây tôi dùng code gọi trực tiếp từ session_state nếu anh đang lưu chunks ở đó
            
            try:
                from retrieval_engine import retrieve_evidence
                evidence = retrieve_evidence(search_query, k=5)
            except ImportError:
                st.error("Không tìm thấy hàm retrieve_evidence. Đang chạy chế độ an toàn (Safe Mode).")
                evidence = [] # Tránh sập app nếu chưa có hàm tìm kiếm

            if not evidence:
                st.warning(f"Không tìm thấy bằng chứng nào trong Ngân hàng tài liệu cho từ khóa: '{search_query}'. Anh hãy nạp thêm PDF ở Tab 1 nhé!")
                return

            # 2. Logic Lắp ráp văn bản (Template Engine)
            draft_text = f"### {section_title}\n\n"
            draft_text += f"Kết quả rà soát y văn cho thấy các dữ liệu quan trọng liên quan đến *{search_query}* như sau:\n\n"
            
            references = []
            
            for idx, ev in enumerate(evidence, start=1):
                # Rút trích thông tin
                text_chunk = ev.get("text", "").strip().replace("\n", " ")
                if len(text_chunk) > 400:
                    text_chunk = text_chunk[:400] + "..." # Rút gọn nếu quá dài
                    
                meta = ev.get("metadata", {})
                file_name = meta.get("file_name", "Tài liệu không tên")
                page = meta.get("page", "N/A")
                
                # Tạo tag trích dẫn [REF-x]
                ref_tag = f"[REF-{idx:03d}]"
                references.append(f"- **{ref_tag}**: {file_name} (Trang {page})")
                
                # Lắp ráp vào câu văn
                draft_text += f"- Theo tài liệu ghi nhận: {text_chunk} {ref_tag}.\n"
                
            draft_text += f"\n*Kết luận:* Các dữ kiện trên cung cấp cơ sở đối chiếu cho việc phân tích đề tài nghiên cứu.\n"

            # 3. Hiển thị kết quả
            st.success("Đã hoàn tất lắp ráp bản thảo!")
            
            st.markdown("---")
            st.markdown(draft_text)
            
            st.markdown("---")
            st.markdown("### Danh mục Trích dẫn gốc")
            st.markdown("\n".join(references))
            
            # Nút copy (nếu cần copy vào Word)
            st.download_button(
                label="📥 Tải Bản thảo (TXT)",
                data=draft_text + "\n\n### Danh mục Trích dẫn gốc\n" + "\n".join(references),
                file_name=f"Ban_thao_{section_title}.txt",
                mime="text/plain"
            )
