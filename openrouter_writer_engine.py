import time
import streamlit as st
from openai import OpenAI
from key_manager import get_next_or_key

def render_openrouter_writer_tab():
    st.markdown("### 🌐 Trợ lý Viết Luận văn (OpenRouter - Tự động thông minh)")
    st.caption("Tự động định tuyến mô hình miễn phí và gán số thứ tự tài liệu tham khảo chuẩn y khoa.")

    if "or_messages" not in st.session_state:
        st.session_state["or_messages"] = []

    # 1. TỰ ĐỘNG THU THẬP VÀ ĐÁNH SỐ THỨ TỰ TÀI LIỆU
    with st.expander("🔍 Dữ liệu bối cảnh và danh mục tham khảo đang nạp", expanded=False):
        context_blocks = []
        ref_counter = 1
        
        evidence = st.session_state.get("last_evidence", [])
        if evidence:
            ev_lines = []
            for e in evidence[:10]:
                ev_lines.append(f"[{ref_counter}] {e.get('text', '')}")
                ref_counter += 1
            context_blocks.append(f"DANH MỤC TÀI LIỆU Y VĂN (RAG):\n" + "\n".join(ev_lines))
            
        summary = st.session_state.get("cached_summary", "")
        if summary:
            context_blocks.append(f"TÓM TẮT ĐỀ TÀI [{ref_counter}]:\n{summary}")
            ref_counter += 1

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

    # 3. XỬ LÝ LỆNH GỌI AI
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
                    with st.spinner(f"🔄 Đang xử lý toàn bộ văn bản (Lần thử {attempt + 1}/{max_retries})..."):
                        client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=current_key,
                            timeout=45.0, # Bảo vệ 45s sẽ hoạt động hiệu quả khi tắt stream
                        )
                        
                        # Gọi API NHƯNG TẮT STREAM
                        response = client.chat.completions.create(
                            model="openrouter/free", 
                            messages=api_messages,
                            temperature=0.2,
                            max_tokens=1024,
                            stream=False # QUAN TRỌNG: Tắt stream để chống treo vĩnh viễn
                        )
                    
                    # Lấy kết quả trả về một lần
                    response_text = response.choices[0].message.content
                    
                    if not response_text:
                        raise ValueError("Máy chủ API trả về kết quả rỗng.")
                        
                    # In kết quả ra màn hình
                    st.markdown(response_text)
                    
                    st.session_state["or_messages"].append({"role": "assistant", "content": response_text})
                    is_success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"⏳ Cổng API nghẽn, đang thử lại... ({e})")
                        time.sleep(2)
                    else:
                        st.error(f"❌ Các mô hình miễn phí hiện đang quá tải. Vui lòng đợi vài phút rồi thử lại. Chi tiết: {e}")
                        if st.session_state["or_messages"]:
                            st.session_state["or_messages"].pop()
