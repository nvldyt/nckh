# File: chat_writing_engine.py (Trợ lý Gemini tự do cho Tab 3)
import os
import streamlit as st
import google.generativeai as genai

def render_writing_chat():
    st.write("---")
    st.subheader("💬 Trợ lý Gemini (Chat tự do)")
    st.caption("Giao diện chat y hệt web Gemini. Dùng để lên ý tưởng, viết lại câu chữ, hoặc nhờ AI giải thích các khái niệm y khoa.")
    
    # 1. Khởi tạo bộ nhớ cho cuộc trò chuyện (Dùng tên biến khác để không đụng hàng với Tab 4)
    if "writing_chat_history" not in st.session_state:
        st.session_state.writing_chat_history = []
        
    # Nút xóa lịch sử chat
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🧹 Xóa hội thoại", key="clear_writing_chat", use_container_width=True):
            st.session_state.writing_chat_history = []
            st.rerun()

    # 2. Hiển thị lịch sử chat trên giao diện
    for message in st.session_state.writing_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Khung nhập liệu Chat
    if prompt := st.chat_input("Nhắn với Gemini để viết hoặc sửa bài... (VD: Hãy viết lại đoạn văn trên cho trôi chảy hơn)"):
        # Hiển thị câu hỏi
        st.session_state.writing_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 4. Giao tiếp với Gemini
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Gemini đang đọc và suy nghĩ..."):
                try:
                    # --- BỘ DÒ TÌM API KEY ĐA TẦNG ---
                    api_key = ""
                    
                    # 1. Tìm trong bộ nhớ tạm (Session State) nếu người dùng vừa nhập
                    if st.session_state.get("GEMINI_API_KEY"):
                        api_key = st.session_state.get("GEMINI_API_KEY")
                    elif st.session_state.get("gemini_api_key"):
                        api_key = st.session_state.get("gemini_api_key")
                        
                    # 2. Tìm trong cấu hình bảo mật của Streamlit Cloud (Secrets)
                    if not api_key:
                        try:
                            if "GEMINI_API_KEY" in st.secrets:
                                api_key = st.secrets["GEMINI_API_KEY"]
                        except:
                            pass
                            
                    # 3. Tìm trong biến môi trường của máy chủ (Environment Variables)
                    if not api_key:
                        api_key = os.environ.get("GEMINI_API_KEY", "")
                        
                    # Nếu vẫn không tìm thấy thì báo lỗi và dừng lại
                    if not api_key:
                        st.error("⚠️ Không tìm thấy Gemini API Key. Vui lòng chuyển sang Tab Cài đặt để nhập Key.")
                        return
                    # ---------------------------------
                        
                    genai.configure(api_key=api_key)
                    # Dùng mô hình Gemini 3.7 Flast
                    model = genai.GenerativeModel("gemini-3.7-flast")
                    
                    # Bơm một câu lệnh mồi (System Prompt) ẩn để Gemini nhập vai xuất sắc hơn
                    system_prompt = "Bạn là một Giáo sư y khoa hướng dẫn sinh viên viết luận văn. Hãy trả lời học thuật, chính xác và chuyên nghiệp.\n\n"
                    
                    # Gộp lịch sử chat để AI nhớ ngữ cảnh
                    chat_context = system_prompt + "Lịch sử trò chuyện:\n"
                    for msg in st.session_state.writing_chat_history[:-1]:
                        chat_context += f"{msg['role'].upper()}: {msg['content']}\n"
                    
                    final_prompt = f"{chat_context}\nCâu hỏi của người dùng: {prompt}"
                    
                    # Gọi AI sinh văn bản
                    response = model.generate_content(final_prompt)
                    full_res = response.text
                    
                    # In ra giao diện và lưu vào bộ nhớ
                    message_placeholder.markdown(full_res)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối AI: {e}")
