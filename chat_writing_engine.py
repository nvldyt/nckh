# File: chat_writing_engine.py
import os
import io
import fitz  # PyMuPDF
import docx  # python-docx
import pandas as pd
import streamlit as st
import google.generativeai as genai

# Hàm đọc chữ từ PDF, Word, Excel
def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            return "\n".join([p.text for p in doc.paragraphs])
        elif file_name.endswith(('.xlsx', '.xls', '.csv')):
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df.to_markdown()
        else:
            return uploaded_file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Lỗi trích xuất dữ liệu: {e}"

def render_writing_chat():
    st.write("---")
    st.subheader("💬 Viết luận văn cùng Gemini")
    st.caption("Chat tự do hoặc đính kèm tài liệu (PDF, Word, Excel...) để AI phân tích.")
    
    if "writing_chat_history" not in st.session_state:
        st.session_state.writing_chat_history = []
        
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🧹 Xóa hội thoại", key="clear_writing_chat", use_container_width=True):
            st.session_state.writing_chat_history = []
            st.rerun()

    # KHAY ĐÍNH KÈM TÀI LIỆU (Sẽ hiện ra ở đây)
    with st.expander("📎 Bấm vào đây để đính kèm tài liệu cho AI đọc", expanded=False):
        uploaded_doc = st.file_uploader("Hỗ trợ PDF, Word, Excel, CSV, TXT", type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'txt'], key="chat_uploader_tab3")
        if uploaded_doc:
            if st.button("📥 Nạp file này vào bộ nhớ AI", use_container_width=True, type="primary"):
                with st.spinner(f"Đang đọc và giải mã {uploaded_doc.name}..."):
                    file_content = extract_text_from_file(uploaded_doc)
                    
                    st.session_state.writing_chat_history.append({
                        "role": "user", 
                        "content": f"[HỆ THỐNG] Người dùng vừa tải lên tài liệu '{uploaded_doc.name}'. Nội dung:\n\n{file_content}"
                    })
                    st.session_state.writing_chat_history.append({
                        "role": "assistant",
                        "content": f"✅ Tôi đã đọc xong file **{uploaded_doc.name}**. Anh cần tôi làm gì với tài liệu này?"
                    })
                    st.rerun()

    # Hiển thị lịch sử chat
    for message in st.session_state.writing_chat_history:
        if "[HỆ THỐNG]" not in message["content"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Khung nhập liệu
    if prompt := st.chat_input("Nhắn với Gemini để viết, sửa bài, hoặc hỏi về file vừa nạp..."):
        st.session_state.writing_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Gemini đang đọc..."):
                try:
                    # Bộ dò tìm API Key
                    api_key = (st.session_state.get("GEMINI_API_KEY") or 
                               st.session_state.get("gemini_api_key") or 
                               os.environ.get("GEMINI_API_KEY", ""))
                    if not api_key:
                        try:
                            api_key = st.secrets.get("GEMINI_API_KEY", "")
                        except: pass
                        
                    if not api_key:
                        st.error("⚠️ Không tìm thấy Gemini API Key.")
                        return
                        
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-pro")
                    
                    system_prompt = "Bạn là một Giáo sư y khoa hướng dẫn sinh viên viết luận văn. Hãy trả lời học thuật, chính xác và chuyên nghiệp.\n\n"
                    
                    chat_context = system_prompt + "Lịch sử trò chuyện:\n"
                    for msg in st.session_state.writing_chat_history[:-1]:
                        chat_context += f"{msg['role'].upper()}: {msg['content']}\n"
                    
                    final_prompt = f"{chat_context}\nCâu hỏi của người dùng: {prompt}"
                    
                    response = model.generate_content(final_prompt)
                    full_res = response.text
                    
                    message_placeholder.markdown(full_res)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối AI: {e}")
