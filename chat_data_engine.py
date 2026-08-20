# File: chat_data_engine.py (Hệ thống Trợ lý Chat Độc lập)
import streamlit as st
import google.generativeai as genai

def render_chat_assistant():
    st.write("---")
    st.subheader("🤖 Trợ lý AI Phân tích Dữ liệu (Lõi Gemini)")
    st.caption("Trò chuyện trực tiếp với dữ liệu Excel của anh. Trợ lý đã tự động đọc tên cột và hiểu cấu trúc dữ liệu.")
    
    # 1. Khởi tạo bộ nhớ cho cuộc trò chuyện
    if "data_chat_history" not in st.session_state:
        st.session_state.data_chat_history = []
        
    # Nút xóa lịch sử chat
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🧹 Xóa hội thoại", use_container_width=True):
            st.session_state.data_chat_history = []
            st.rerun()

    # 2. Hiển thị lịch sử chat trên giao diện
    for message in st.session_state.data_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Khung nhập liệu Chat (Hoạt động như ChatGPT)
    if prompt := st.chat_input("Hỏi AI cách xử lý dữ liệu, biểu đồ, hoặc code SPSS..."):
        # Hiển thị câu hỏi của người dùng
        st.session_state.data_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # KHÂU THẦN KỲ: Rút trích dữ liệu Excel bí mật gửi cho AI
        context = ""
        if "excel_data" in st.session_state and st.session_state["excel_data"] is not None and not st.session_state["excel_data"].empty:
            df = st.session_state["excel_data"]
            cols = ", ".join(df.columns.astype(str).tolist())
            shape = df.shape
            # Lấy 3 dòng đầu tiên làm mẫu cho AI xem (dùng Markdown để AI dễ đọc)
            sample_data = df.head(3).to_markdown() 
            
            context = f"""
[BỐI CẢNH ẨN - KHÔNG IN RA MÀN HÌNH]
Người dùng đang mở một file Excel với thông tin sau:
- Tổng số: {shape[0]} dòng, {shape[1]} cột.
- Tên các cột: {cols}
- Dữ liệu mẫu (3 dòng đầu):
{sample_data}
Hãy dựa vào cấu trúc dữ liệu này để đưa ra câu trả lời chính xác, sát thực tế nhất cho câu hỏi dưới đây.
"""
        
        # 4. Giao tiếp với Gemini
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI đang suy nghĩ và kiểm tra dữ liệu..."):
                try:
                    # Tự động quét tìm API Key ở mọi ngóc ngách trong session_state của ứng dụng chính
                    api_key = (
                        st.session_state.get("GEMINI_API_KEY") or 
                        st.session_state.get("gemini_api_key") or 
                        st.session_state.get("API_KEY") or 
                        ""
                    )

                    if not api_key:
                        st.error("⚠️ Không tìm thấy Gemini API Key. Vui lòng nhập ở Tab Cài đặt.")
                        return
                        
                    genai.configure(api_key=api_key)
                    # Dùng mô hình mạnh nhất hiện tại để phân tích số liệu
                    model = genai.GenerativeModel("gemini-3.7-flast")
                    
                    # Gộp lịch sử chat để AI nhớ ngữ cảnh
                    chat_context = "Lịch sử trò chuyện trước đó:\n"
                    for msg in st.session_state.data_chat_history[:-1]:
                        chat_context += f"{msg['role'].upper()}: {msg['content']}\n"
                    
                    # Ghép lệnh cuối cùng: Lịch sử + Dữ liệu Excel + Câu hỏi mới
                    final_prompt = f"{chat_context}\n{context}\nCâu hỏi hiện tại của người dùng: {prompt}"
                    
                    response = model.generate_content(final_prompt)
                    full_res = response.text
                    
                    # Hiển thị và lưu lại
                    message_placeholder.markdown(full_res)
                    st.session_state.data_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối AI: {e}")
