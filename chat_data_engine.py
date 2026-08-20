# File: chat_data_engine.py (Hệ thống Trợ lý Chat Độc lập - Chống bịa dữ liệu)
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
    st.caption("Trò chuyện trực tiếp với dữ liệu Excel của anh. AI đã bị ép buộc bám sát danh sách thuốc thực tế.")
    
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

    # 4. Khung nhập liệu Chat
    if prompt := st.chat_input("Hỏi AI cách xử lý dữ liệu, biểu đồ, hoặc phân tích tương tác thuốc..."):
        st.session_state.data_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # KHÂU THẦN KỲ MỚI: Rút trích danh sách thực tế để ép AI học thuộc
        context = ""
        if "excel_data" in st.session_state and st.session_state["excel_data"] is not None and not st.session_state["excel_data"].empty:
            df = st.session_state["excel_data"]
            cols = ", ".join(df.columns.astype(str).tolist())
            shape = df.shape
            
            # --- Thu thập dữ liệu thực tế chống bịa đặt ---
            unique_info = ""
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    unique_vals = df[col].dropna().astype(str).unique()
                    # Lấy danh sách nếu cột có dưới 250 loại giá trị khác nhau (tránh tràn token)
                    if 0 < len(unique_vals) <= 250: 
                        val_str = ", ".join(unique_vals)
                        unique_info += f"- Cột '{col}' chứa CHÍNH XÁC các giá trị này: {val_str}\n"

            sample_data = df.head(3).to_markdown() 
            
            context = f"""
[BỐI CẢNH ẨN - HƯỚNG DẪN NGHIÊM NGẶT]
Người dùng đang phân tích file Excel y khoa với {shape[0]} dòng và {shape[1]} cột.
Tên các cột: {cols}

DANH SÁCH GIÁ TRỊ THỰC TẾ ĐANG CÓ TRONG FILE (Dùng để đối chiếu):
{unique_info}

LỆNH BẮT BUỘC DÀNH CHO AI:
1. TUYỆT ĐỐI KHÔNG BỊA ĐẶT (hallucinate) tên thuốc, thảo dược hoặc số liệu không có trong "Danh sách giá trị thực tế" ở trên.
2. Khi phân tích tương tác thuốc hoặc lập bảng, CHỈ ĐƯỢC PHÉP sử dụng các tên thuốc/hoạt chất có xuất hiện thực tế trong danh sách.
3. Nếu người dùng hỏi về một thông tin/thuốc không có trong dữ liệu, phải trả lời rõ: "Dữ liệu thực tế trong file không chứa loại thuốc này".
4. Dữ liệu mẫu (3 dòng đầu) để hiểu cấu trúc: 
{sample_data}
"""
        
        # 5. Giao tiếp với Gemini
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI đang quét dữ liệu thực tế và phân tích..."):
                try:
                    safety_settings = {
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                    }
                    
                    # Dùng mô hình Gemini 3.7 Flast
                    model = genai.GenerativeModel("gemini-3.7-flast")
                    
                    # Giới hạn lịch sử để chống lỗi 429
                    recent_history = st.session_state.data_chat_history[-8:]
                    chat_context = "Lịch sử trò chuyện gần đây:\n"
                    for msg in recent_history[:-1]:
                        content_snippet = msg['content'][:800] if len(msg['content']) > 800 else msg['content']
                        chat_context += f"{msg['role'].upper()}: {content_snippet}\n"
                    
                    final_prompt = f"{chat_context}\n{context}\nCâu hỏi hiện tại của người dùng: {prompt}"
                    
                    # Cơ chế xoay vòng Key
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
                                continue 
                            else:
                                raise inner_e

                    full_res = response.text
                    
                    # Hiển thị và lưu lại
                    message_placeholder.markdown(full_res)
                    st.session_state.data_chat_history.append({"role": "assistant", "content": full_res})
                    
                except Exception as e:
                    error_msg = f"❌ Hệ thống quá tải hoặc gặp lỗi. Anh vui lòng bấm nút **Xóa hội thoại** phía trên hoặc thử lại sau 15 giây. Chi tiết: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.data_chat_history.append({"role": "assistant", "content": error_msg})
