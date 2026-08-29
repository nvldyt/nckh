# File: openrouter_writer_engine.py
import time
import streamlit as st
from openai import OpenAI
from key_manager import get_next_or_key

def render_openrouter_writer_tab():
    st.markdown("### 🌐 Trợ lý Viết Luận văn (OpenRouter - Auto Free)")
    st.caption("Tự động định tuyến qua các mô hình nguồn mở mạnh mẽ nhất hiện đang miễn phí.")

    if "or_messages" not in st.session_state:
        st.session_state["or_messages"] = []

    # 1. TỰ ĐỘNG THU THẬP BỐI CẢNH TỪ CÁC TAB 1, 2, 3, 4, 7
    with st.expander("🔍 Dữ liệu bối cảnh đang nạp tự động", expanded=False):
        context_blocks = []
        
        evidence = st.session_state.get("last_evidence", [])
        if evidence:
            ev_text = "\n".join([f"- {e.get('text', '')}" for e in evidence[:5]])
            context_blocks.append(f"TÀI LIỆU Y VĂN (RAG):\n{ev_text}")
            
        summary = st.session_state.get("cached_summary", "")
        if summary:
            context_blocks.append(f"TÓM TẮT ĐỀ TÀI:\n{summary}")

        saved_tables = st.session_state.get("saved_tables", {})
        if saved_tables:
            table_info = "".join([f"Bảng {name}:\n{df.to_markdown()}\n\n" for name, df in saved_tables.items()])
            context_blocks.append(f"BẢNG SỐ LIỆU NGHIÊN CỨU:\n{table_info}")

        compiled_context = "\n\n".join(context_blocks)
        if compiled_context:
            st.success("✅ Đã kết nối và đồng bộ dữ liệu từ các Tab nghiên cứu.")
        else:
            st.info("ℹ️ Chưa có dữ liệu nền nào được nạp từ các Tab trước.")

    # 2. HIỂN THỊ LỊCH SỬ CHAT
    for msg in st.session_state["or_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. XỬ LÝ LỆNH GỌI AI
    user_query = st.chat_input("Yêu cầu AI viết (VD: Phân tích bảng số liệu 1 dựa trên các y văn đã trích xuất)...")

    if user_query:
        st.session_state["or_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            system_instruction = (
                "Bạn là một chuyên gia Dược lâm sàng xuất sắc, hỗ trợ nghiên cứu viên viết luận văn Chuyên khoa I. "
                "Văn phong yêu cầu: Khách quan, khoa học, đi thẳng vào vấn đề, sử dụng đúng thuật ngữ y khoa. "
                "Hãy lập luận dựa trên BẢNG SỐ LIỆU và TÀI LIỆU Y VĂN được cung cấp bên dưới. "
                "Khi phân tích thực trạng sử dụng thuốc hay chế độ liều, hãy gắn liền với điều kiện thực tế tại bệnh viện tuyến tỉnh. "
                "Tuyệt đối không bịa đặt số liệu.\n\n"
                f"=== DỮ LIỆU ĐỀ TÀI ===\n{compiled_context}"
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
                    )
                    
                    # Bổ sung max_tokens để mô hình phản hồi ổn định
                    stream = client.chat.completions.create(
                        model="openrouter/free",
                        messages=api_messages,
                        temperature=0.2,
                        max_tokens=2048,
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
