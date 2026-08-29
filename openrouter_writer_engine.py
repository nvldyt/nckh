import time
import streamlit as st
from openai import OpenAI
from key_manager import get_next_or_key

def render_openrouter_writer_tab():
    st.markdown("**🌐 Trợ lý Viết Luận văn (OpenRouter - Qwen 72B)**")
    st.caption("Tối ưu văn phong học thuật và lập luận logic sâu.")

    if "or_messages" not in st.session_state:
        st.session_state["or_messages"] = []

    # TỰ ĐỘNG ĐỒNG BỘ DỮ LIỆU TỪ CÁC TAB KHÁC
    with st.expander("🔍 Dữ liệu bối cảnh đang nạp", expanded=False):
        context_blocks = []
        
        evidence = st.session_state.get("last_evidence", [])
        if evidence:
            ev_text = "\n".join([f"- {e.get('text', '')}" for e in evidence[:5]])
            context_blocks.append(f"TÀI LIỆU Y VĂN:\n{ev_text}")
            
        saved_tables = st.session_state.get("saved_tables", {})
        if saved_tables:
            table_info = "".join([f"Bảng {name}:\n{df.to_markdown()}\n\n" for name, df in saved_tables.items()])
            context_blocks.append(f"BẢNG SỐ LIỆU:\n{table_info}")

        compiled_context = "\n\n".join(context_blocks)
        if compiled_context:
            st.success("Đã đồng bộ dữ liệu từ Tab PubMed, RAG và Phân tích số liệu.")
        else:
            st.info("Chưa có dữ liệu nền được nạp.")

    # HIỂN THỊ HỘI THOẠI
    for msg in st.session_state["or_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # XỬ LÝ LỆNH GỌI API VỚI CƠ CHẾ THỬ LẠI (RETRY) TỰ ĐỘNG
    user_query = st.chat_input("Yêu cầu AI viết (VD: Phân tích bảng số liệu 1)...")

    if user_query:
        st.session_state["or_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            system_instruction = (
                "Bạn là chuyên gia Dược lâm sàng hỗ trợ viết luận văn Chuyên khoa I. "
                "Hãy lập luận dựa trên BẢNG SỐ LIỆU và Y VĂN được cung cấp. Phân tích cần bám sát điều kiện "
                "thực tế tại Bệnh viện ĐKTP Vinh. Tuyệt đối không bịa đặt số liệu.\n\n"
                f"=== DỮ LIỆU ĐỀ TÀI ===\n{compiled_context}"
            )
            
            api_messages = [{"role": "system", "content": system_instruction}]
            api_messages.extend(st.session_state["or_messages"][-4:])
            
            is_success = False
            max_retries = 3 # Thử tối đa 3 key liên tiếp nếu gặp lỗi
            
            for attempt in range(max_retries):
                current_key = get_next_or_key()
                try:
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=current_key,
                    )
                    
                    stream = client.chat.completions.create(
                        model="qwen/qwen-2.5-72b-instruct:free",
                        messages=api_messages,
                        temperature=0.2,
                        stream=True
                    )
                    
                    response_text = st.write_stream(stream)
                    st.session_state["or_messages"].append({"role": "assistant", "content": response_text})
                    is_success = True
                    break # Thoát vòng lặp nếu thành công
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"🔄 Đang xoay vòng Key OpenRouter do lỗi: {e}")
                        time.sleep(1)
                    else:
                        st.error(f"❌ Toàn bộ Key đều gặp lỗi. Vui lòng thử lại sau. Chi tiết: {e}")
                        st.session_state["or_messages"].pop()
