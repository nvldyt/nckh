# File: chat_writing_engine.py (Bản OFFLINE Tối ưu RAM & Xoay vòng Key)
import os
import io
import fitz  # PyMuPDF
import docx  # python-docx
import pandas as pd
import streamlit as st
import google.generativeai as genai
import gc # Thêm thư viện dọn rác RAM

import key_manager # Bắt buộc gọi trái tim chứa 8 Key ở đây

# Hàm đọc chữ từ PDF, Word, Excel (Đã tối ưu dọn RAM)
def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.pdf'):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
            return text
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([p.text for p in doc.paragraphs])
            del doc # Dọn RAM
            return text
        elif file_name.endswith(('.xlsx', '.xls', '.csv')):
            df = pd.read_csv(uploaded_file) if file_name.endswith('.csv') else pd.read_excel(uploaded_file)
            markdown_str = df.to_markdown()
            del df # Dọn RAM
            return markdown_str
        else:
            return uploaded_file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Lỗi trích xuất dữ liệu: {e}"
    finally:
        gc.collect() # Ép hệ thống nhả RAM ngay lập tức

def render_writing_chat():
    st.write("---")
    st.subheader("💬 Viết luận văn cùng Gemini (Bản Offline)")
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
                    
                    system_prompt = "Bạn là một Giáo sư y khoa hướng dẫn sinh viên viết luận văn. Hãy trả lời học thuật, chính xác và chuyên nghiệp.\n\n"
                    
                    # Lấy 8 tin nhắn gần nhất để chống tràn token
                    recent_history = st.session_state.writing_chat_history[-8:]
                    chat_context = system_prompt + "Lịch sử trò chuyện gần đây:\n"
                    for msg in recent_history[:-1]:
                        content_snippet = msg['content'][:800] if len(msg['content']) > 800 else msg['content']
                        chat_context += f"{msg['role'].upper()}: {content_snippet}\n"
                    
                    final_prompt = f"{chat_context}\nCâu hỏi của người dùng: {prompt}"
                    
                    # Cơ chế gọi API xoay vòng 8 key
                    response = None
                    for attempt in range(2):
                        try:
                            # BƯỚC 1: Lấy key, gọt sạch khoảng trắng và cấu hình TRƯỚC
                            api_key = key_manager.get_next_key().strip()
                            if not api_key:
                                raise ValueError("Không tìm thấy API Key!")
                            genai.configure(api_key=api_key)
                            
                            # BƯỚC 2: Khai báo model SAU KHI đã cấu hình key (Chuẩn tên 1.5-flash)
                            model = genai.GenerativeModel("gemini-1.5-flash") 
                            
                            # BƯỚC 3: Kích hoạt AI
                            response = model.generate_content(final_prompt, safety_settings=safety_settings)
                            break
                        except Exception as inner_e:
                            if ("429" in str(inner_e) or "Quota" in str(inner_e)) and attempt == 0:
                                continue # Đổi sang key tiếp theo và thử lại
                            else:
                                raise inner_e

                    full_res = response.text
                    
                    message_placeholder.markdown(full_res)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    error_msg = f"❌ Hệ thống gặp sự cố: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": error_msg})
                    
                except Exception as e:
                    error_msg = f"❌ Hệ thống gặp sự cố: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.writing_chat_history.append({"role": "assistant", "content": error_msg})
