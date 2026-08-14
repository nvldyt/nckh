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
    
    # 1. NÂNG CẤP SYSTEM PROMPT: ÉP VĂN PHONG KHÔ KHAN, CHUYÊN SÂU
    system_prompt = """
    Bạn là một chuyên gia dược lâm sàng, thống kê y học và biên tập viên luận văn y khoa cấp cao. 
    Quy tắc làm việc của bạn:
   Bạn là một chuyên gia dược lâm sàng, thống kê y học và biên tập viên luận văn y khoa cấp cao. 
    Quy tắc làm việc của bạn:
    1. VĂN PHONG (QUAN TRỌNG NHẤT): Tuyệt đối KHÔNG sử dụng từ ngữ hoa mỹ, bay bổng, sáo rỗng. Văn phong phải cực kỳ khô khan, trực diện, logic chặt chẽ.
    2. TÍNH CHUYÊN SÂU: Sử dụng chính xác 100% thuật ngữ chuyên ngành. Tập trung vào bằng chứng (Evidence-based), cơ sở sinh lý bệnh, cơ chế dược lý, số liệu dịch tễ.
    3. CHỐNG BỊA ĐẶT TUYỆT ĐỐI (ANTI-HALLUCINATION): TUYỆT ĐỐI KHÔNG tự bịa số liệu, kết quả hay bất kỳ thông tin nào. Nếu tài liệu đầu vào KHÔNG có thông tin, BẮT BUỘC phải trả lời: 'Tài liệu không đề cập'. KHÔNG ĐƯỢC lấy dữ liệu bên ngoài Internet để đắp vào.
    4. Trích dẫn: Mỗi khẳng định bắt buộc kèm theo [Tên tác giả, Năm]. 
    5. Bảng biểu: Trình bày dưới dạng bảng Markdown chuẩn.
    """
    
    model = genai.GenerativeModel("gemini-3.6-flash", system_instruction=system_prompt)
    
    # 2. THIẾT LẬP NHIỆT ĐỘ (TEMPERATURE) = 0.1 ĐỂ DIỆT SỰ SÁNG TẠO/BAY BỔNG
    generation_config = genai.types.GenerationConfig(
        temperature=0.1,
    )
    
except Exception as e:
    st.error("Chưa cấu hình API Key trong Streamlit Secrets. Vui lòng thêm khóa vào mục cài đặt của ứng dụng.")
    st.stop()

# ==========================================
# CÁC TAB CHỨC NĂNG
# ==========================================
tab1, tab2 = st.tabs(["📄 Đọc Tài liệu & Viết Luận văn (PDF)", "📊 Phân tích Số liệu Bệnh án (Excel)"])

# ----------------------------------------------------
# TAB 1: PHÂN TÍCH TÀI LIỆU THAM KHẢO
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
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            reader = PdfReader(uploaded_file)
            file_content = ""
            for page in reader.pages:
                file_content += page.extract_text() + "\n"
            combined_text += f"\n--- TÀI LIỆU {index}: {uploaded_file.name} ---\n{file_content}\n"
        
        st.success(f"Đã đọc thành công {len(uploaded_files)} tài liệu PDF!")
        
        st.write("---")
        st.subheader("📝 Lệnh viết nhanh cho luận văn (Bấm là chạy):")
        
        citation_rules = """
        QUY TẮC TRÍCH DẪN, HÀN LÂM & CHỐNG BỊA ĐẶT BẮT BUỘC:
        1. Bất kỳ câu khẳng định số liệu, dịch tễ nào cũng PHẢI có trích dẫn số trong ngoặc vuông (VD: [1], [2, 3]) ở cuối câu.
        2. Các số trích dẫn phải theo thứ tự xuất hiện liên tục.
        3. KIỂM CHỨNG SỐ LIỆU: Khi trích dẫn các số liệu quan trọng (tỷ lệ %, p-value...), yêu cầu giữ nguyên văn ý nghĩa của bản gốc. Nếu không có số liệu, ghi rõ "Các tài liệu cung cấp không đề cập".
        4. Cuối đoạn văn bản, BẮT BUỘC liệt kê "Tài liệu tham khảo" chi tiết tương ứng với các số đã dùng.
        """

       # Tạo 5 cột cho 5 nút bấm đúng theo tiến trình luận văn
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("Viết Đặt vấn đề"):
                with st.spinner("AI đang viết phần Đặt vấn đề..."):
                    prompt = f"Dựa trên các tài liệu PDF, hãy viết phần 'Đặt vấn đề' cho luận văn CKI Dược lâm sàng, tập trung vào số liệu dịch tễ, tính cấp thiết lâm sàng và khoảng trống nghiên cứu. {citation_rules}"
                    full_prop = f"Cơ sở dữ liệu:\n{combined_text}\n\nYêu cầu: {prompt}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
                    st.markdown(response.text)
                    
        with col2:
            if st.button("Viết Tổng quan"):
                with st.spinner("AI đang viết phần Tổng quan..."):
                    prompt = f"Viết phần tổng quan y văn chuyên sâu, phân tích cơ chế sinh lý bệnh, dược lý và phác đồ điều trị dựa trên các file PDF. {citation_rules}"
                    full_prop = f"Cơ sở dữ liệu:\n{combined_text}\n\nYêu cầu: {prompt}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
                    st.markdown(response.text)
                    
        with col3:
            if st.button("Viết Phương pháp NC"):
                with st.spinner("AI đang thiết kế Chương 2: Phương pháp nghiên cứu..."):
                    prompt = f"""
                    Dựa trên phương pháp luận của các tài liệu PDF, hãy viết "Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU" cho luận văn.
                    BẮT BUỘC tuân thủ NGHIÊM NGẶT cấu trúc sau:
                    2.1. Đối tượng, thời gian, địa điểm nghiên cứu (Nêu rõ tiêu chuẩn lựa chọn, loại trừ).
                    2.2. Phương pháp nghiên cứu
                    2.2.1. Thiết kế nghiên cứu.
                    2.2.2. Cỡ mẫu và phương pháp chọn mẫu.
                    2.2.3. Chỉ tiêu nghiên cứu (BẮT BUỘC trình bày dưới dạng Bảng Markdown gồm 5 cột: TT | Tên chỉ tiêu/Biến số | Định nghĩa/Giải thích | Phân loại biến | Kỹ thuật thu thập).
                    2.2.4. Phương pháp thu thập số liệu.
                    2.2.5. Xử lý và phân tích số liệu (Chia làm 2 tiểu mục: 2.2.5.1. Xử lý số liệu; 2.2.5.2. Phân tích số liệu - Nêu rõ phần mềm và các test thống kê).
                    
                    Văn phong khô khan, chuyên sâu. {citation_rules}
                    """
                    full_prop = f"Cơ sở dữ liệu:\n{combined_text}\n\nYêu cầu: {prompt}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
                    st.markdown(response.text)
                    
        with col4:
            if st.button("Viết Bàn luận"):
                with st.spinner("AI đang viết phần Bàn luận..."):
                    prompt = f"Viết phần bàn luận y khoa chuyên sâu: so sánh kết quả (p-value, tỷ lệ), giải thích nguyên nhân khác biệt dựa trên dược động học/dược lực học, và nêu hạn chế nghiên cứu. {citation_rules}"
                    full_prop = f"Cơ sở dữ liệu:\n{combined_text}\n\nYêu cầu: {prompt}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
                    st.markdown(response.text)
                    
        with col5:
            if st.button("Trích dẫn Vancouver"):
                with st.spinner("AI đang lập danh mục tài liệu..."):
                    prompt = "Lập danh mục tài liệu tham khảo định dạng Vancouver chuẩn xác từ các tài liệu trên."
                    full_prop = f"Cơ sở dữ liệu:\n{combined_text}\n\nYêu cầu: {prompt}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
                    st.markdown(response.text)
       st.write("---")
        custom_prompt = st.text_area("Hoặc tự nhập yêu cầu riêng của anh cho toàn bộ tập tài liệu:")      
        if st.button("Chạy lệnh tùy chỉnh"):
            if custom_prompt:
                with st.spinner("AI đang xử lý yêu cầu..."):
                    # Tự động nhúng Lớp 3 và Lớp 4 vào mọi yêu cầu tự do của anh
                    anti_hallucination_spell = """
                    \nLƯU Ý NGHIÊM NGẶT: Tuyệt đối không tự bịa thông tin. Nếu trong tài liệu PDF KHÔNG có số liệu hoặc thông tin tôi hỏi, bắt buộc phải trả lời: 'Tài liệu không đề cập'. Yêu cầu BẮT BUỘC trích dẫn lại NGUYÊN VĂN (Copy - Paste) câu văn chứa số liệu đó trong tài liệu PDF gốc và đặt trong dấu ngoặc kép ("...") để tôi kiểm chứng.
                    """
                    full_prop = f"Cơ sở dữ liệu:\n{combined_text}\n\nYêu cầu: {custom_prompt}\n{anti_hallucination_spell}\n{citation_rules}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
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
        df = pd.read_excel(excel_file)
        columns = df.columns.tolist()
        
        st.success(f"Đọc dữ liệu thành công! File có {df.shape[0]} dòng và {df.shape[1]} cột.")
        
        with st.expander("👀 Xem trước dữ liệu gốc"):
            st.dataframe(df.head())

        st.write("---")
        st.subheader("🛠️ CÔNG CỤ PHÂN TÍCH CHUYÊN SÂU")
        
        # --- TÍNH NĂNG 1: THỐNG KÊ TẦN SỐ (AUTO FREQUENCIES) ---
        st.markdown("### 1. Thống kê mô tả (Auto Frequencies)")
        st.info("💡 Mẹo: Chọn nhiều biến cùng lúc để AI tự động đếm tần số và nhận xét hàng loạt. KHÔNG chọn biến định danh (Họ tên, Số bệnh án...).")
        
        vars_desc = st.multiselect("🏷️ Chọn TẤT CẢ các biến cần thống kê (VD: Giới tính, Nhóm tuổi, Mức độ bệnh...):", columns, key="auto_var_desc")
        
        if st.button("🚀 Chạy toàn bộ Thống kê & Nhận xét"):
            if not vars_desc:
                st.warning("Vui lòng chọn ít nhất 1 biến để phân tích!")
            else:
                for var in vars_desc:
                    with st.spinner(f"AI đang tính toán tần số cho biến '{var}'..."):
                        # Dùng Python đếm số liệu thực tế
                        freq_table = df[var].value_counts().to_string()
                        total = len(df[var].dropna())
                        
                        prompt = f"""
                        Dữ liệu đếm thực tế của biến '{var}': {freq_table} (Tổng: {total}).
                        Yêu cầu: 1. Vẽ bảng SPSS (Phân loại, n, %). 2. Viết nhận xét y khoa chuyên sâu, khô khan, không bay bổng.
                        """
                        response = model.generate_content(prompt, generation_config=generation_config)
                        
                        # In kết quả ra màn hình
                        st.subheader(f"► Phân tích biến: {var}")
                        st.markdown(response.text)
                        st.write("---") # Đường kẻ ngang phân cách
        st.write("---")
        
        st.markdown("### 2. Bảng chéo & Phân tích mối liên quan (Auto Crosstabs)")
        st.info("💡 Mẹo: Chọn 1 Biến phụ thuộc (Cột) làm gốc. Sau đó chọn nhiều Biến độc lập (Hàng) để AI tự động chạy hàng loạt các bảng. KHÔNG chọn biến định danh (Họ tên, Số bệnh án...).")
        
        target_col = st.selectbox("🎯 Chọn Biến Phụ thuộc / Cột (VD: Mức độ bệnh, Kết quả điều trị):", columns, key="auto_target")
        indep_cols = st.multiselect("🏷️ Chọn TẤT CẢ các Biến Độc lập / Hàng (VD: Giới tính, Nhóm tuổi, Tiền sử...):", columns, key="auto_indep")
            
        if st.button("🚀 Chạy toàn bộ Bảng chéo & Nhận xét"):
            if not indep_cols:
                st.warning("Vui lòng chọn ít nhất 1 biến độc lập (hàng) ở ô phía trên!")
            else:
                for var in indep_cols:
                    if var == target_col:
                        continue # Bỏ qua nếu anh lỡ chọn trùng 2 biến giống nhau
                        
                    with st.spinner(f"AI đang xử lý bảng chéo giữa '{var}' và '{target_col}'..."):
                        crosstab_df = pd.crosstab(df[var], df[target_col])
                        
                        prompt = f"""
                        Bảng Crosstabs thực tế giữa '{var}' và '{target_col}':\n{crosstab_df.to_string()}\n
                        Yêu cầu: 1. Trình bày bảng khoa học (% hàng/cột). 2. Nhận xét cực kỳ chuyên sâu về y học, tuyệt đối không dùng từ ngữ cảm xúc.
                        """
                        response = model.generate_content(prompt, generation_config=generation_config)
                        
                        # In ra màn hình từng phần rõ ràng
                        st.subheader(f"► Mối liên quan giữa {var} và {target_col}")
                        st.markdown(response.text)
                        st.write("---") # Đường kẻ ngang phân cách giữa các bảng
        st.write("---")
        
        st.markdown("### 3. Trợ lý AI tự do")
        analysis_prompt = st.text_area("Nhập yêu cầu (VD: Nhận xét tổng quan):")
        
        if st.button("Chạy lệnh tùy chỉnh"):
            if analysis_prompt:
                with st.spinner("AI đang xử lý..."):
                    desc_stats = df.describe(include='all').to_string()
                    data_prompt = f"Bảng thống kê tổng quát:\n{desc_stats}\n\nYêu cầu: {analysis_prompt}. Viết văn phong hàn lâm, khô khan."
                    response = model.generate_content(data_prompt, generation_config=generation_config)
                    st.markdown(response.text)
            else:
                st.warning("Vui lòng nhập yêu cầu!")
