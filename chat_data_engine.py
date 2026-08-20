# File: chat_data_engine.py (Hệ thống Trợ lý Chat Độc lập)
import streamlit as st
import os
import io
import fitz  # PyMuPDF
import docx  # python-docx
import pandas as pd
import google.generativeai as genai

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
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
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
        if st.button("🧹 Xóa hội thoại", use_container_width=True):
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
        # Ẩn bớt các tin nhắn hệ thống dài dòng để giao diện gọn gàng
        if "[HỆ THỐNG]" not in message["content"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 4. Khung nhập liệu Chat (Hoạt động như ChatGPT)
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
Người dùng đang mở một file Excel chính với thông tin sau:
- Tổng số: {shape[0]} dòng, {shape[1]} cột.
- Tên các cột: {cols}
- Dữ liệu mẫu (3 dòng đầu):
{sample_data}
Hãy dựa vào cấu trúc dữ liệu này để đưa ra câu trả lời chính xác, sát thực tế nhất cho câu hỏi dưới đây.
"""
        
        # 5. Giao tiếp với Gemini
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI đang suy nghĩ và kiểm tra dữ liệu..."):
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
                        
                    if not api_key:
                        st.error("⚠️ Không tìm thấy Gemini API Key. Vui lòng nhập ở Tab Cài đặt (Tab cuối cùng).")
                        return
                    # ---------------------------------
                        
                    genai.configure(api_key=api_key)
                    
                    # Dùng mô hình Gemini 1.5 Pro chuẩn của Google API
                    model = genai.GenerativeModel("gemini-1.5-pro")
                    
                    # Gộp lịch sử chat để AI nhớ ngữ cảnh
                    chat_context = "Lịch sử trò chuyện trước đó:\n"
                    for msg in st.session_state.data_chat_history[:-1]:
                        chat_context += f"{msg['role'].upper()}: {msg['content']}\n"
                    
                    # Ghép lệnh cuối cùng: Lịch sử + Dữ liệu Excel (nếu có) + Câu hỏi mới
                    final_prompt = f"{chat_context}\n{context}\nCâu hỏi hiện tại của người dùng: {prompt}"
                    
                    response = model.generate_content(final_prompt)
                    full_res = response.text
                    
                    # Hiển thị và lưu lại
                    message_placeholder.markdown(full_res)
                    st.session_state.data_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối AI: {e}")
