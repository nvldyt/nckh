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

    # Khung nhập liệu (Đã tối ưu hóa chống tràn token và xoay vòng 8 key)
    if prompt := st.chat_input("Nhắn với Gemini để viết, sửa bài, hoặc hỏi về file vừa nạp..."):
        st.session_state.writing_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Gemini đang suy nghĩ và tổng hợp..."):
                try:
                    import key_manager  # Nạp module quản lý 8 key
                    
                    safety_settings = {
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                    }
                    
                    # Dùng mô hình flash tốc độ cao để tránh nghẽn
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    system_prompt = "Bạn là một Giáo sư y khoa hướng dẫn sinh viên viết luận văn. Hãy trả lời học thuật, chính xác và chuyên nghiệp.\n\n"
                    
                    # Chỉ lấy tối đa 8 tin nhắn gần nhất để tiết kiệm token tối đa, không bị lỗi 429
                    recent_history = st.session_state.writing_chat_history[-8:]
                    chat_context = system_prompt + "Lịch sử trò chuyện gần đây:\n"
                    for msg in recent_history[:-1]:
                        # Cắt gọn các đoạn nội dung quá dài trong lịch sử
                        content_snippet = msg['content'][:800] if len(msg['content']) > 800 else msg['content']
                        chat_context += f"{msg['role'].upper()}: {content_snippet}\n"
                    
                    final_prompt = f"{chat_context}\nCâu hỏi của người dùng: {prompt}"
                    
                    # Cơ chế tự động đổi Key nếu gặp lỗi quá tải
                    response = None
                    for attempt in range(2): # Thử tối đa 2 lần với các key khác nhau nếu lỗi
                        try:
                            api_key = key_manager.get_next_key()
                            genai.configure(api_key=api_key)
                            response = model.generate_content(final_prompt, safety_settings=safety_settings)
                            break
                        except Exception as inner_e:
                            if "429" in str(inner_e) and attempt == 0:
                                continue # Lập tức đổi sang key kế tiếp và thử lại
                            else:
                                raise inner_e

                    full_res = response.text
                    
                    # Hiển thị và lưu lại
                    message_placeholder.markdown(full_res)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    error_msg = f"❌ Hệ thống đang quá tải (429). Anh vui lòng bấm nút **Xóa hội thoại** phía trên hoặc đợi 15 giây rồi hỏi lại nhé. Chi tiết: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": error_msg})
