import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

st.title("Trợ lý Nghiên cứu & Viết Luận văn")

api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.5-flash")

    uploaded_file = st.file_uploader("Tải lên tài liệu nghiên cứu (PDF)", type="pdf")

    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        st.success("Đã đọc xong tài liệu!")

        prompt = st.text_area("Bạn muốn viết gì từ tài liệu này? (Ví dụ: Viết phần Đặt vấn đề)")

        if st.button("Chạy lệnh"):
            with st.spinner("AI đang tổng hợp và viết..."):
                full_prop = f"Dựa vào tài liệu sau đây:\n{text}\n\nHãy thực hiện yêu cầu: {prompt}"
                response = model.generate_content(full_prop)
                st.write(response.text)
else:
    st.warning("Vui lòng nhập API Key ở thanh bên trái để bắt đầu.")
