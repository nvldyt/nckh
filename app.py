import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd

st.set_page_config(page_title="Trợ lý NCKH Dược Lâm Sàng", layout="wide")

st.title("HỖ TRỢ NGHIÊN CỨU KHOA HỌC")

# Tự động lấy API Key từ bộ nhớ bảo mật của Streamlit Cloud
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    system_prompt = """
    Bạn là một chuyên gia dược lâm sàng và biên tập viên luận văn y khoa cấp cao. 
    Quy tắc làm việc của bạn:
    1. Văn phong: Học thuật, khách quan, chính xác, sử dụng thuật ngữ y khoa chuẩn (tương đương tạp chí Y học Việt Nam/Quốc tế).
    2. Viết nhận xét: Phải so sánh kết quả của bệnh nhân với các nghiên cứu trong tài liệu PDF (nếu có) hoặc với các hướng dẫn điều trị chuẩn (như KDIGO, GINA, GOLD).
    3. Trích dẫn: Mỗi khi đưa ra khẳng định, bắt buộc phải kèm theo [Tên tác giả, Năm]. 
    4. Bảng biểu: Kết quả phải được trình bày dưới dạng bảng Markdown chuẩn.
    5. Bàn luận: Phải phân tích sâu sắc tại sao số liệu lại như vậy, không chỉ liệt kê số.
    """
    model = genai.GenerativeModel("gemini-3.1-pro")
except Exception as e:
    st.error("Chưa cấu hình API Key trong Streamlit Secrets. Vui lòng thêm khóa vào mục cài đặt của ứng dụng.")
    st.stop()

# --- CÁC TAB CHỨC NĂNG ---
tab1, tab2 = st.tabs(["📄 Đọc Y văn & Viết Luận văn (PDF)", "📊 Phân tích Số liệu Bệnh án (Excel)"])

with tab1:
    st.header("Trợ lý tổng hợp tài liệu y văn")
    
    uploaded_files = st.file_uploader(
        "Tải lên nhiều tài liệu nghiên cứu (PDF)", 
        type="pdf", 
        accept_multiple_files=True, 
        key="pdf_uploader"
    )
    
    if uploaded_files:
        combined_text = ""
        for uploaded_file in uploaded_files:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                combined_text += page.extract_text() + "\n"
        
        st.success(f"Đã đọc thành công {len(uploaded_files)} tài liệu PDF!")
        
        # Đưa các nút bấm lên TRÊN để luôn hiển thị ngay lập tức
        st.write("---")
        st.subheader("📝 Lệnh viết nhanh cho luận văn (Bấm là chạy):")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Viết Đặt vấn đề"):
                with st.spinner("AI đang viết phần Đặt vấn đề..."):
                    prompt = "Dựa trên các tài liệu được cung cấp, hãy viết phần 'Đặt vấn đề' cho luận văn CKI Dược lâm sàng, nêu bật tính cấp thiết, ý nghĩa khoa học và mục tiêu nghiên cứu. Sử dụng văn phong học thuật, khách quan."
                    full_prop = f"Tổng hợp tài liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
                    
        with col2:
            if st.button("Viết Tổng quan"):
                with st.spinner("AI đang viết phần Tổng quan..."):
                    prompt = "Viết phần tổng quan tài liệu dựa trên tất cả các file PDF được cung cấp. Sử dụng văn phong học thuật, khách quan, trích dẫn đầy đủ."
                    full_prop = f"Tổng hợp tài liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
                    
        with col3:
            if st.button("Viết Bàn luận"):
                with st.spinner("AI đang viết phần Bàn luận..."):
                    prompt = "Dựa trên các tài liệu này, hãy viết phần bàn luận: so sánh kết quả nghiên cứu, giải thích cơ chế sinh lý bệnh và nêu rõ hạn chế."
                    full_prop = f"Tổng hợp tài liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
                    
        with col4:
            if st.button("Trích dẫn Vancouver"):
                with st.spinner("AI đang lập danh mục tài liệu tham khảo chuẩn Vancouver..."):
                    prompt = "Từ các tài liệu y văn được cung cấp, hãy lập danh mục tài liệu tham khảo được định dạng chính xác theo chuẩn Vancouver (Số thứ tự [1], [2]... theo mẫu: Tác giả AA, Tác giả BB. Tên bài báo. Tên tạp chí viết tắt Năm;Tập(Số):Trang)."
                    full_prop = f"Tổng hợp tài liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
        
        st.write("---")
        custom_prompt = st.text_area("Hoặc tự nhập yêu cầu riêng của anh cho toàn bộ tập tài liệu:")
        if st.button("Chạy lệnh tùy chỉnh"):
            if custom_prompt:
                with st.spinner("AI đang xử lý yêu cầu..."):
                    full_prop = f"Tổng hợp tài liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {custom_prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
            else:
                st.warning("Vui lòng nhập yêu cầu vào ô trống!")
                
        # Thu gọn danh sách file trong khung bấm mở rộng để giao diện không bị dài trôi trang
        with st.expander("📂 Xem danh sách các file PDF đã tải lên"):
            for f in uploaded_files:
                st.text(f"- {f.name} ({round(f.size / 1024, 1)} KB)")
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
