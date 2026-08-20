# File: chat_data_engine.py (Hệ thống Trợ lý Chat Độc lập - Đã tối ưu hóa)
import streamlit as st
import os
import io
import itertools
import fitz  # PyMuPDF
import docx  # python-docx
import pandas as pd
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
    return os.environ.get("GEMINI_API_KEY", "")
# ----------------------------------------------------

# Hàm phụ trợ: Đọc và trích xuất chữ từ nhiều loại file
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

def render_chat_assistant():
    st.write("---")
    st.subheader("🤖 Phân tích Dữ liệu với Gemini")
    st.caption("Trò chuyện trực tiếp với dữ liệu Excel của anh. AI đã tự động đọc tên cột và hiểu cấu trúc dữ liệu.")
    
    # 1. Khởi tạo bộ nhớ cho cuộc trò chuyện
    if "data_chat_history" not in st.session_state:
        st.session_state.data_chat_history = []
        
    # Nút xóa lịch sử chat
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🧹 Xóa hội thoại", key="clear_data_chat", use_container_width=True):
            st.session_state.data_chat_history = []
            st.rerun()

    # ==========================================
    # 2. KHAY ĐÍNH KÈM TÀI LIỆU CHO TAB 4
    # ==========================================
    with st.expander("📎 Bấm vào đây để đính kèm thêm tài liệu cho AI đọc (nếu cần)", expanded=False):
        uploaded_doc = st.file_uploader("Hỗ trợ PDF, Word, Excel, CSV, TXT", type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'txt'], key="chat_uploader_tab4")
        if uploaded_doc:
            if st.button("📥 Nạp file này vào bộ nhớ AI", use_container_width=True, type="primary", key="btn_load_tab4"):
                with st.spinner(f"Đang đọc và giải mã {uploaded_doc.name}..."):
                    file_content = extract_text_from_file(uploaded_doc)
                    
                    st.session_state.data_chat_history.append({
                        "role": "user", 
                        "content": f"[HỆ THỐNG] Người dùng vừa tải lên tài liệu phụ '{uploaded_doc.name}'. Dưới đây là nội dung:\n\n{file_content}"
                    })
                    st.session_state.data_chat_history.append({
                        "role": "assistant",
                        "content": f"✅ Tôi đã đọc và ghi nhớ toàn bộ nội dung file **{uploaded_doc.name}**. Anh cần tôi phân tích gì với tài liệu này?"
                    })
                    st.rerun()

    # 3. Hiển thị lịch sử chat trên giao diện
    for message in st.session_state.data_chat_history:
        if "[HỆ THỐNG]" not in message["content"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 4. Khung nhập liệu Chat (Hoạt động như ChatGPT)
    if prompt := st.chat_input("Hỏi AI cách xử lý dữ liệu, biểu đồ, hoặc code SPSS..."):
        st.session_state.data_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # KHÂU THẦN KỲ: Rút trích dữ liệu Excel bí mật gửi cho AI
        context = ""
        if "excel_data" in st.session_state and st.session_state["excel_data"] is not None and not st.session_state["excel_data"].empty:
            df = st.session_state["excel_data"]
            cols = ", ".join(df.columns.astype(str).tolist())
            shape = df.shape
            sample_data = df.head(3).to_markdown() 
            
            context = f"""
[BỐI CẢNH ẨN - KHÔNG IN RA MÀN HÌNH]
Người dùng đang mở một file Excel chính với thông tin sau:
- Tổng số: {shape[0]} dòng, {shape[1]} cột.
- Tên các cột: {cols}
- Dữ liệu mẫu (3 dòng đầu):
{sample_data}
Hãy dựa vào cấu trúc dữ liệu này để đưa ra câu trả lời chính xác, sát thực tế nhất cho câu hỏi dưới đây.
"""
        
        # 5. Giao tiếp với Gemini (Đã tối ưu xoay vòng Key và chống tràn token)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI đang suy nghĩ và kiểm tra dữ liệu..."):
                try:
                    safety_settings = {
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                    }
                    
                    # Dùng mô hình Gemini 1.5 Flash tốc độ cao
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    # Chỉ lấy tối đa 8 tin nhắn gần nhất để tiết kiệm token, tránh lỗi 429
                    recent_history = st.session_state.data_chat_history[-8:]
                    chat_context = "Lịch sử trò chuyện gần đây:\n"
                    for msg in recent_history[:-1]:
                        content_snippet = msg['content'][:800] if len(msg['content']) > 800 else msg['content']
                        chat_context += f"{msg['role'].upper()}: {content_snippet}\n"
                    
                    final_prompt = f"{chat_context}\n{context}\nCâu hỏi hiện tại của người dùng: {prompt}"
                    
                    # Cơ chế tự động đổi Key và thử lại khi gặp lỗi giới hạn
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
                                continue # Tự động chuyển sang key kế tiếp và thử lại
                            else:
                                raise inner_e

                    full_res = response.text
                    
                    # Hiển thị và lưu lại
                    message_placeholder.markdown(full_res)
                    st.session_state.data_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    error_msg = f"❌ Hệ thống quá tải hoặc hết hạn mức (429). Anh vui lòng bấm nút **Xóa hội thoại** phía trên hoặc đợi 15 giây rồi hỏi lại nhé. Chi tiết: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.data_chat_history.append({"role": "assistant", "content": error_msg})
