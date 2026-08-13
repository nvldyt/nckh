import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd

st.set_page_config(page_title="Trợ lý NCKH Dược Lâm Sàng", layout="wide")

st.title("Trợ lý Nghiên cứu & Phân tích Dữ liệu Y khoa")

# Tự động lấy API Key từ bộ nhớ bảo mật của Streamlit Cloud
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.5-flash")
except Exception as e:
    st.error("Chưa cấu hình API Key trong Streamlit Secrets. Vui lòng thêm khóa vào mục cài đặt của ứng dụng.")
    st.stop()

# --- CÁC TAB CHỨC NĂNG ---
tab1, tab2 = st.tabs(["📄 Đọc Y văn & Viết Luận văn (PDF)", "📊 Phân tích Số liệu Bệnh án (Excel)"])

with tab1:
    st.header("Trợ lý tổng hợp tài liệu y văn")
    uploaded_file = st.file_uploader("Tải lên tài liệu nghiên cứu (PDF)", type="pdf", key="pdf_uploader")
    
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        st.success("Đã đọc xong tài liệu PDF!")
        prompt = st.text_area("Bạn muốn viết gì từ tài liệu này?")
        
        if st.button("Chạy lệnh viết văn"):
            with st.spinner("AI đang tổng hợp và viết..."):
                full_prop = f"Dựa vào tài liệu sau đây:\n{text}\n\nHãy thực hiện yêu cầu: {prompt}"
                response = model.generate_content(full_prop)
                st.write(response.text)

with tab2:
    st.header("Phân tích thống kê mô tả số liệu")
    excel_file = st.file_uploader("Tải lên file số liệu bệnh án (Excel .xlsx)", type="xlsx", key="excel_uploader")
    
    if excel_file is not None:
        df = pd.read_excel(excel_file)
        st.subheader("1. Xem trước 5 dòng dữ liệu đầu tiên:")
        st.dataframe(df.head())
        
        st.subheader("2. Bảng thống kê mô tả tự động:")
        desc_stats = df.describe(include='all')
        st.dataframe(desc_stats)
        
        analysis_prompt = st.text_area("Nhập yêu cầu phân tích số liệu:")
        
        if st.button("Chạy phân tích số liệu bằng AI"):
            with st.spinner("AI đang đọc bảng số liệu và viết báo cáo..."):
                stats_string = desc_stats.to_string()
                data_prompt = f"Dưới đây là bảng thống kê mô tả số liệu nghiên cứu:\n{stats_string}\n\nYêu cầu: {analysis_prompt}."
                res = model.generate_content(data_prompt)
                st.write(res.text)
