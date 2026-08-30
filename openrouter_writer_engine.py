# File: openrouter_writer_engine.py
import time
import streamlit as st
from openai import OpenAI
from key_manager import get_next_or_key

def render_openrouter_writer_tab():
    st.markdown("### 🌐 Trợ lý Viết Luận văn (OpenRouter - Trích dẫn số tự động)")
    st.caption("Tự động định tuyến mô hình mở và gán số thứ tự tài liệu tham khảo chuẩn y khoa.")

    if "or_messages" not in st.session_state:
        st.session_state["or_messages"] = []

    # 1. TỰ ĐỘNG THU THẬP VÀ ĐÁNH SỐ THỨ TỰ TÀI LIỆU TỪ CÁC TAB 1, 2, 3, 4, 7
    with st.expander("🔍 Dữ liệu bối cảnh và danh mục tham khảo đang nạp", expanded=False):
        context_blocks = []
        ref_counter = 1
        
        # Đánh số thứ tự cho RAG / PubMed (Tab 1 & 2)
        evidence = st.session_state.get("last_evidence", [])
        if evidence:
            ev_lines = []
            for e in evidence[:10]: # Lấy tối đa 10 tài liệu để đánh số
                ev_lines.append(f"[{ref_counter}] {e.get('text', '')}")
                ref_counter += 1
            context_blocks.append(f"DANH MỤC TÀI LIỆU Y VĂN (RAG):\n" + "\n".join(ev_lines))
            
        # Đánh số thứ tự cho tóm tắt hoặc tài liệu khác (Tab 4)
        summary = st.session_state.get("cached_summary", "")
        if summary:
            context_blocks.append(f"TÓM TẮT ĐỀ TÀI [{ref_counter}]:\n{summary}")
            ref_counter += 1

        # Nạp bảng số liệu (Tab 7)
        saved_tables = st.session_state.get("saved_tables", {})
        if saved_tables:
            table_info = "".join([f"Bảng {name}:\n{df.to_markdown()}\n\n" for name, df in saved_tables.items()])
            context_blocks.append(f"BẢNG SỐ LIỆU NGHIÊN CỨU:\n{table_info}")

        compiled_context = "\n\n".join(context_blocks)
        if compiled_context:
            st.success(f"✅ Đã đồng bộ dữ liệu và thiết lập sơ đồ đánh số trích dẫn (Tổng số nguồn: {ref_counter - 1}).")
        else:
            st.info("ℹ️ Chưa có dữ liệu nền nào được nạp từ các Tab trước.")

    # 2. HIỂN THỊ LỊCH SỬ CHAT
    for msg in st.session_state["or_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. XỬ LÝ LỆNH GỌI AI VỚI QUY TẮC TRÍCH DẪN SỐ
    user_query = st.chat_input("Yêu cầu AI viết (VD: Viết phần bàn luận và đính kèm số trích dẫn [1], [2] tương ứng)...")

    if user_query:
        st.session_state["or_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            system_instruction = (
                "Bạn là một chuyên gia Dược lâm sàng xuất sắc, hỗ trợ nghiên cứu viên viết luận văn Chuyên khoa I. "
                "YÊU CẦU BẮT BUỘC VỀ TRÍCH DẪN: "
                "1. Khi sử dụng thông tin, số liệu, hoặc kết luận từ các tài liệu được cung cấp, bạn PHẢI đính kèm số thứ tự tài liệu tham khảo dạng ngoặc vuông ở cuối câu (ví dụ: [1], [2]). "
                "2. Các số trích dẫn phải tuân thủ đúng thứ tự xuất hiện của nguồn tài liệu trong ngữ cảnh bên dưới. "
                "3. Tuyệt đối không bịa đặt số liệu hoặc tự ý gán nguồn sai sự thật. "
                "4. Văn phong: Khách quan, khoa học, chuẩn mực y khoa, gắn liền thực tế bệnh viện tuyến tỉnh.\n\n"
                f"=== DỮ LIỆU ĐỀ TÀI VÀ NGUỒN THAM KHẢO ===\n{compiled_context}"
            )
            
            api_messages = [{"role": "system", "content": system_instruction}]
            api_messages.extend(st.session_state["or_messages"][-4:])
            
            is_success = False
            max_retries = 3
            
            for attempt in range(max_retries):
                current_key = get_next_or_key()
                try:
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=current_key,
                        timeout=30.0, # Thêm thời gian chờ tối đa 30 giây để tránh treo đơ
                    )
                    
                    stream = client.chat.completions.create(
                        model="model="nvidia/llama-3.1-nemotron-70b-instruct:free",
                        messages=api_messages,
                        temperature=0.2,
                        max_tokens=1024, # Giảm xuống 1024 để sinh nhanh hơn, tránh nghẽn
                        stream=True
                    )
                    
                    response_text = st.write_stream(stream)
                    st.session_state["or_messages"].append({"role": "assistant", "content": response_text})
                    is_success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"🔄 Đang chuyển key tiếp theo do lỗi: {e}")
                        time.sleep(1)
                    else:
                        st.error(f"❌ Toàn bộ Key OpenRouter đều gặp lỗi. Chi tiết: {e}")
                        st.session_state["or_messages"].pop()
