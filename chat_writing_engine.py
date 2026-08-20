# File: chat_writing_engine.py
import os
import io
import itertools
import fitz  # PyMuPDF
import docx  # python-docx
import pandas as pd
import streamlit as st
import google.generativeai as genai

# --- CƠ CHẾ XOAY VÒNG 8 KEY TRỰC TIẾP TỪ SECRETS ---
@st.cache_resource
def get_key_cycler():
    try:
        raw_keys = st.secrets.get("GEMINI_API_KEY", "")
        if raw_keys:
            keys_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
            if keys_list:
                return itertools.cycle(keys_list)
    except Exception:
        pass
    return None

key_cycle = get_key_cycler()

def get_next_api_key():
    if key_cycle:
        return next(key_cycle)
    # Fallback nếu chạy local lấy từ biến môi trường
    return os.environ.get("GEMINI_API_KEY", "")
# ----------------------------------------------------

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
            df = pd.read_csv(uploaded_file) if file_name.endswith('.csv') else pd.read_excel(uploaded_file)
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

    # KHAY ĐÍNH KÈM TÀI LIỆU
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
            with st.spinner("Gemini đang suy nghĩ và tổng hợp..."):
                try:
                    safety_settings = {
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                    }
                    
                    model = genai.GenerativeModel("gemini-3.7-flash")
                    
                    system_prompt = "Bạn là một Giáo sư y khoa hướng dẫn sinh viên viết luận văn. Hãy trả lời học thuật, chính xác và chuyên nghiệp.\n\n"
                    
                    # Lấy 8 tin nhắn gần nhất để chống tràn token
                    recent_history = st.session_state.writing_chat_history[-8:]
                    chat_context = system_prompt + "Lịch sử trò chuyện gần đây:\n"
                    for msg in recent_history[:-1]:
                        content_snippet = msg['content'][:800] if len(msg['content']) > 800 else msg['content']
                        chat_context += f"{msg['role'].upper()}: {content_snippet}\n"
                    
                    final_prompt = f"{chat_context}\nCâu hỏi của người dùng: {prompt}"
                    
                    # Cơ chế gọi API xoay vòng 8 key từ Secrets và tự động đổi nếu dính quá tải (429)
                    response = None
                    for attempt in range(2):
                        try:
                            api_key = get_next_api_key()
                            if not api_key:
                                raise ValueError("Không tìm thấy API Key trong Secrets!")
                            genai.configure(api_key=api_key)
                            
                            response = model.generate_content(final_prompt, safety_settings=safety_settings)
                            break
                        except Exception as inner_e:
                            if ("429" in str(inner_e) or "Quota" in str(inner_e)) and attempt == 0:
                                continue # Đổi sang key tiếp theo trong chuỗi và thử lại ngay
                            else:
                                raise inner_e

                    full_res = response.text
                    
                    message_placeholder.markdown(full_res)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    error_msg = f"❌ Hệ thống gặp sự cố: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": error_msg})
