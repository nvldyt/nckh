import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN (NỀN XANH TÍM + FONT ARIAL AN TOÀN)
# ==========================================
st.set_page_config(page_title="NCKH", layout="wide")

custom_css = """
<style>
    /* Ép phông chữ Arial an toàn (Không làm hỏng icon tải file) */
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li {
        font-family: 'Arial', sans-serif;
    }

    /* Nền toàn bộ trang web (Gradient xanh dương - tím nhạt) */
    .stApp {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
    }
    
    /* Chỉnh màu và hiệu ứng cho Tiêu đề chính */
    h1 {
        color: #4a148c !important; 
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-weight: 800;
        margin-bottom: 30px;
    }

    /* Nền trắng mờ cho các Tab để dễ đọc chữ trên nền màu */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* Trang trí thanh Tab */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
        padding: 5px;
    }

    /* Làm đẹp các nút bấm (Button) - Màu tím đậm */
    div.stButton > button {
        background-color: #6a1b9a !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    /* Hiệu ứng khi di chuột vào nút bấm */
    div.stButton > button:hover {
        background-color: #4a148c !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Tùy chỉnh bảng dữ liệu Dataframe */
    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("HỖ TRỢ NGHIÊN CỨU KHOA HỌC")

# ==========================================
# CẤU HÌNH API GEMINI 3.6 FLASH
# ==========================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    system_prompt = """
    Bạn là một chuyên gia dược lâm sàng, thống kê y học và biên tập viên luận văn y khoa cấp cao. 
    Quy tắc làm việc của bạn:
    1. Văn phong: Học thuật, khách quan, chính xác, sử dụng thuật ngữ y khoa chuẩn (tương đương tạp chí Y học Việt Nam/Quốc tế).
    2. Viết nhận xét: Phải so sánh kết quả của bệnh nhân với các nghiên cứu trong tài liệu PDF (nếu có) hoặc với các hướng dẫn điều trị chuẩn.
    3. Trích dẫn: Mỗi khi đưa ra khẳng định, bắt buộc phải kèm theo [Tên tác giả, Năm]. 
    4. Bảng biểu: Kết quả phải được trình bày dưới dạng bảng Markdown chuẩn (mô phỏng chuẩn bảng của SPSS).
    5. Bàn luận: Phải phân tích sâu sắc tại sao số liệu lại như vậy, không chỉ liệt kê số.
    """
    
    # Khai báo sử dụng lõi gemini-3.6-flash
    model = genai.GenerativeModel("gemini-3.6-flash", system_instruction=system_prompt)
    
except Exception as e:
    st.error("Chưa cấu hình API Key trong Streamlit Secrets. Vui lòng thêm khóa vào mục cài đặt của ứng dụng.")
    st.stop()

# ==========================================
# CÁC TAB CHỨC NĂNG
# ==========================================
tab1, tab2 = st.tabs(["📄 Đọc Y văn & Viết Luận văn (PDF)", "📊 Phân tích Số liệu Bệnh án (Excel)"])

# ----------------------------------------------------
# TAB 1: PHÂN TÍCH TÀI LIỆU Y VĂN (NÂNG CẤP TRÍCH DẪN)
# ----------------------------------------------------
with tab1:
    st.header("Phân tích tài liệu và Viết bài")
    
    uploaded_files = st.file_uploader(
        "Tải lên nhiều tài liệu nghiên cứu (PDF)", 
        type="pdf", 
        accept_multiple_files=True, 
        key="pdf_uploader"
    )
    
    if uploaded_files:
        combined_text = ""
        # Trích xuất văn bản và đính kèm tên file để AI biết nguồn
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            reader = PdfReader(uploaded_file)
            file_content = ""
            for page in reader.pages:
                file_content += page.extract_text() + "\n"
            combined_text += f"\n--- TÀI LIỆU {index}: {uploaded_file.name} ---\n{file_content}\n"
        
        st.success(f"Đã đọc thành công {len(uploaded_files)} tài liệu PDF!")
        
        st.write("---")
        st.subheader("📝 Lệnh viết nhanh cho luận văn (Bấm là chạy):")
        
        # Bổ sung câu lệnh gốc (Core Prompt) ép buộc trích dẫn Vancouver nghiêm ngặt
        citation_rules = """
        QUY TẮC TRÍCH DẪN BẮT BUỘC (CHUẨN VANCOUVER):
        1. Bất kỳ một câu khẳng định số liệu, dịch tễ, hay kết luận y khoa nào trong văn bản cũng PHẢI có trích dẫn ở cuối câu bằng số đặt trong ngoặc vuông (VD: [1], [2, 3]).
        2. Các số trích dẫn phải được đánh số theo đúng thứ tự xuất hiện liên tục trong đoạn văn bản bạn viết ra (Bắt đầu từ [1] cho tài liệu đầu tiên được nhắc đến, [2] cho tài liệu tiếp theo...). Không được nhảy cóc số.
        3. Tuyệt đối không tự bịa (hallucinate) thông tin. Chỉ sử dụng số liệu/kiến thức từ các Tài liệu PDF được cung cấp.
        4. Cuối đoạn văn bản, BẮT BUỘC phải tạo một danh sách "Tài liệu tham khảo" chi tiết tương ứng với các số [1], [2]... bạn vừa dùng trong bài.
        """

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Viết Đặt vấn đề"):
                with st.spinner("AI đang viết phần Đặt vấn đề..."):
                    prompt = f"""
                    Dựa trên các tài liệu PDF được cung cấp, hãy viết phần 'Đặt vấn đề' cho luận văn CKI Dược lâm sàng, nêu bật tính cấp thiết, ý nghĩa khoa học và mục tiêu nghiên cứu. 
                    {citation_rules}
                    """
                    full_prop = f"Cơ sở dữ liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
                    
        with col2:
            if st.button("Viết Tổng quan"):
                with st.spinner("AI đang viết phần Tổng quan..."):
                    prompt = f"""
                    Viết phần tổng quan tài liệu (Tổng quan y văn) một cách logic, mạch lạc dựa trên tất cả các file PDF được cung cấp. 
                    {citation_rules}
                    """
                    full_prop = f"Cơ sở dữ liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
                    
        with col3:
            if st.button("Viết Bàn luận"):
                with st.spinner("AI đang viết phần Bàn luận..."):
                    prompt = f"""
                    Dựa trên các tài liệu này, hãy viết phần bàn luận: so sánh các kết quả nghiên cứu, giải thích cơ chế sinh lý bệnh, nguyên nhân của sự khác biệt số liệu và nêu rõ hạn chế.
                    {citation_rules}
                    """
                    full_prop = f"Cơ sở dữ liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
                    
        with col4:
            if st.button("Trích dẫn Vancouver"):
                with st.spinner("AI đang lập danh mục tài liệu tham khảo chuẩn Vancouver..."):
                    prompt = "Từ các tài liệu y văn được cung cấp, hãy lập danh mục tài liệu tham khảo tổng hợp được định dạng chính xác theo chuẩn Vancouver (Tác giả AA, Tác giả BB. Tên bài báo. Tên tạp chí viết tắt Năm;Tập(Số):Trang)."
                    full_prop = f"Cơ sở dữ liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {prompt}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
        
        st.write("---")
        custom_prompt = st.text_area("Hoặc tự nhập yêu cầu riêng của anh cho toàn bộ tập tài liệu:")
        if st.button("Chạy lệnh tùy chỉnh"):
            if custom_prompt:
                with st.spinner("AI đang xử lý yêu cầu..."):
                    full_prop = f"Cơ sở dữ liệu y văn:\n{combined_text}\n\nYêu cầu của tôi: {custom_prompt}\n{citation_rules}"
                    response = model.generate_content(full_prop)
                    st.markdown(response.text)
            else:
                st.warning("Vui lòng nhập yêu cầu vào ô trống!")
                
        with st.expander("📂 Xem danh sách các file PDF đã tải lên"):
            for f in uploaded_files:
                st.text(f"- {f.name} ({round(f.size / 1024, 1)} KB)")
# ----------------------------------------------------
# TAB 2: PHÂN TÍCH SỐ LIỆU TỪ EXCEL (MÔ PHỎNG SPSS)
# ----------------------------------------------------
with tab2:
    st.header("📊 Phân tích thống kê & Kiểm định (Chuẩn SPSS từ Excel)")
    
    excel_file = st.file_uploader("Tải lên file số liệu bệnh án (Excel .xlsx)", type=["xlsx", "xls"], key="excel_uploader")
    
    if excel_file is not None:
        # Đọc dữ liệu từ file Excel
        df = pd.read_excel(excel_file)
        columns = df.columns.tolist()
        
        st.success(f"Đọc dữ liệu thành công! File có {df.shape[0]} dòng và {df.shape[1]} cột.")
        
        with st.expander("👀 Xem trước dữ liệu gốc"):
            st.dataframe(df.head())

        st.write("---")
        st.subheader("🛠️ CÔNG CỤ PHÂN TÍCH CHUYÊN SÂU")
        
        # --- TÍNH NĂNG 1: THỐNG KÊ TẦN SỐ (FREQUENCIES) ---
        st.markdown("### 1. Thống kê mô tả (Frequencies)")
        var_desc = st.selectbox("Chọn biến cần thống kê (VD: Giới tính, Nhóm tuổi, Mức độ bệnh):", columns, key="var_desc")
        
        if st.button("Chạy Thống kê & Nhận xét"):
            with st.spinner(f"AI đang tính toán tần số cho biến {var_desc}..."):
                freq_table = df[var_desc].value_counts().to_string()
                total = len(df[var_desc].dropna())
                
                prompt = f"""
                Dưới đây là số liệu đếm thực tế của biến '{var_desc}' từ file Excel:
                {freq_table}
                (Tổng số mẫu hợp lệ: {total})
                
                Yêu cầu:
                1. Hãy vẽ lại một bảng chuẩn format SPSS bao gồm các cột: Phân loại, Tần số (n), Tỷ lệ (%).
                2. Viết một đoạn nhận xét y khoa chuyên nghiệp dựa trên các con số trong bảng này (Dùng cho luận văn CKI).
                """
                response = model.generate_content(prompt)
                st.markdown(response.text)

        st.write("---")
        
        # --- TÍNH NĂNG 2: BẢNG CHÉO & MỐI LIÊN QUAN (CROSSTABS) ---
        st.markdown("### 2. Bảng chéo & Phân tích mối liên quan (Crosstabs)")
        st.info("💡 Tính năng này mô phỏng Crosstabs của SPSS để xem xét mối liên quan giữa 2 biến (VD: Tuổi và Mức độ nặng).")
        
        col_a, col_b = st.columns(2)
        with col_a:
            var_row = st.selectbox("Biến Độc lập / Hàng (VD: Giới tính):", columns, key="var_row")
        with col_b:
            var_col = st.selectbox("Biến Phụ thuộc / Cột (VD: Mức độ bệnh):", columns, key="var_col")
            
        if st.button("Chạy Crosstabs & Nhận xét"):
            with st.spinner("AI đang xử lý bảng chéo..."):
                crosstab_df = pd.crosstab(df[var_row], df[var_col])
                crosstab_str = crosstab_df.to_string()
                
                prompt = f"""
                Dưới đây là bảng chéo (Crosstabs) thực tế trích xuất từ file Excel giữa 2 biến: '{var_row}' và '{var_col}':
                \n{crosstab_str}\n
                Yêu cầu:
                1. Hãy trình bày lại thành một bảng biểu chuẩn khoa học (Có cột Tổng, Hàng Tổng và tính Tỷ lệ % theo hàng hoặc cột sao cho hợp lý).
                2. Đóng vai trò chuyên gia, viết nhận xét về mối phân bố/liên quan giữa '{var_row}' và '{var_col}' dựa hoàn toàn vào các con số thực tế trong bảng trên.
                """
                response = model.generate_content(prompt)
                st.markdown(response.text)

        st.write("---")
        
        # --- TÍNH NĂNG 3: AI TỰ PHÂN TÍCH THEO YÊU CẦU ---
        st.markdown("### 3. Trợ lý AI tự do (Phân tích toàn bộ dữ liệu)")
        analysis_prompt = st.text_area("Nhập câu hỏi hoặc yêu cầu (VD: Viết nhận xét tổng quan về độ tuổi và các bệnh mắc kèm của tập dữ liệu này):")
        
        if st.button("Chạy lệnh tùy chỉnh"):
            if analysis_prompt:
                with st.spinner("AI đang xử lý..."):
                    desc_stats = df.describe(include='all').to_string()
                    data_prompt = f"Đây là bảng thống kê tổng quát của file Excel:\n{desc_stats}\n\nYêu cầu của tôi: {analysis_prompt}"
                    
                    response = model.generate_content(data_prompt)
                    st.markdown(response.text)
            else:
                st.warning("Vui lòng nhập yêu cầu!")
