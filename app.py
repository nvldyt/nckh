import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd

st.set_page_config(page_title="Trợ lý NCKH Dược Lâm Sàng", layout="wide")

st.title("Trợ lý Nghiên cứu & Phân tích Dữ liệu Y khoa")

# Nhập API Key ở thanh bên trái
api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Tạo các Tab phân chia chức năng rõ ràng
    tab1, tab2 = st.tabs(["📄 Đọc Y văn & Viết Luận văn (PDF)", "📊 Phân tích Số liệu Bệnh án (Excel)"])

    # --- TAB 1: ĐỌC TÀI LIỆU PDF ---
    with tab1:
        st.header("Trợ lý tổng hợp tài liệu y văn")
        uploaded_file = st.file_uploader("Tải lên tài liệu nghiên cứu (PDF)", type="pdf", key="pdf_uploader")
        
        if uploaded_file is not None:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            st.success("Đã đọc xong tài liệu PDF!")
            
            prompt = st.text_area("Bạn muốn viết gì từ tài liệu này? (VD: Viết phần Tổng quan tài liệu, Đặt vấn đề...)")
            
            if st.button("Chạy lệnh viết văn"):
                with st.spinner("AI đang tổng hợp và viết..."):
                    full_prop = f"Dựa vào tài liệu sau đây:\n{text}\n\nHãy thực hiện yêu cầu: {prompt}"
                    response = model.generate_content(full_prop)
                    st.write(response.text)

    # --- TAB 2: PHÂN TÍCH SỐ LIỆU EXCEL ---
    with tab2:
        st.header("Phân tích thống kê mô tả số liệu")
        excel_file = st.file_uploader("Tải lên file số liệu bệnh án (Excel .xlsx)", type="xlsx", key="excel_uploader")
        
        if excel_file is not None:
            # Đọc file Excel bằng thư viện pandas
            df = pd.read_excel(excel_file)
            
            st.subheader("1. Xem trước 5 dòng dữ liệu đầu tiên:")
            st.dataframe(df.head())
            
            st.subheader("2. Bảng thống kê mô tả tự động (Mean, Min, Max, Tần số...):")
            # Tự động tính toán các chỉ số thống kê mô tả
            desc_stats = df.describe(include='all')
            st.dataframe(desc_stats)
            
            st.subheader("3. Yêu cầu AI viết báo cáo kết quả dựa trên số liệu:")
            analysis_prompt = st.text_area("Nhập yêu cầu phân tích (VD: Hãy viết đoạn nhận xét về tuổi và cân nặng của bệnh nhân dựa trên bảng thống kê trên)")
            
            if st.button("Chạy phân tích số liệu bằng AI"):
                with st.spinner("AI đang đọc bảng số liệu và viết báo cáo..."):
                    stats_string = desc_stats.to_string()
                    data_prompt = f"Dưới đây là bảng thống kê mô tả số liệu nghiên cứu dược lâm sàng:\n{stats_string}\n\nYêu cầu của tôi: {analysis_prompt}. Hãy viết thành đoạn văn bản hoàn chỉnh, chuẩn văn phong khoa học y tế cho luận văn CKI."
                    res = model.generate_content(data_prompt)
                    st.write(res.text)
else:
    st.warning("Vui lòng nhập Google Gemini API Key ở thanh bên trái để bắt đầu sử dụng ứng dụng.")
