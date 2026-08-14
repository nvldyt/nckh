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
# TAB 1: PHÂN TÍCH TÀI LIỆU Y VĂN & NGÂN HÀNG DỮ LIỆU
# ----------------------------------------------------
with tab1:
    st.header("Phân tích tài liệu và Viết bài")
    
    # Khởi tạo Bộ nhớ vĩnh cửu cho ứng dụng
    if "ngan_hang_y_van" not in st.session_state:
        st.session_state["ngan_hang_y_van"] = ""
        
    st.markdown("### 🏦 Tổng hợp tóm tắt các nghiên cứu")
    st.info("💡 Mẹo: Tải từng đợt 2-3 bài báo, bấm nút 'Rút trích' để AI hút số liệu lưu vào đây. Sau đó xóa bài cũ, tải bài mới lên rút trích tiếp cho đến khi đủ tài liệu.")
    
    # Ô chứa dữ liệu cộng dồn
    st.session_state["ngan_hang_y_van"] = st.text_area(
        "Dữ liệu trích xuất từ các nghiên cứu (Có thể tự bổ sung):", 
        st.session_state["ngan_hang_y_van"], 
        height=200
    )
    
    st.write("---")
    
    uploaded_files = st.file_uploader(
        "Tải lên tài liệu nghiên cứu (PDF) để AI đọc:", 
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
        
        if st.button("📥 Rút trích số liệu & Ghi vào Ngân hàng", type="primary"):
            with st.spinner("AI đang vắt kiệt thông tin từ các file PDF và lưu vào bộ nhớ..."):
                extract_prompt = """
                Hãy đọc các tài liệu PDF trên và TÓM TẮT CÔ ĐẶC lại những thông tin sau:
                1. Tên tác giả, năm nghiên cứu, tên bài báo.
                2. Mục tiêu nghiên cứu và đối tượng nghiên cứu.
                3. Các kết quả, số liệu quan trọng nhất (tỷ lệ %, p-value, OR, RR...).
                4. Kết luận chính của tác giả.
                Tuyệt đối không bịa số liệu. Trình bày dưới dạng gạch đầu dòng ngắn gọn.
                """
                full_prop = f"Tài liệu gốc:\n{combined_text}\n\nYêu cầu: {extract_prompt}"
                response = model.generate_content(full_prop, generation_config=generation_config)
                
                # Cộng dồn dữ liệu mới vào dữ liệu cũ
                st.session_state["ngan_hang_y_van"] += f"\n\n{response.text}"
                st.rerun() # Tải lại trang để cập nhật ô Text Area
                
    st.write("---")
    
    # --- TRẠM TRUNG CHUYỂN DỮ LIỆU TỪ TAB 2 SANG TAB 1 ---
    st.markdown("### 🌉 Bộ nhớ Số liệu của riêng bạn (Dành cho phần Bàn luận)")
    my_research_data = st.text_area(
        "Copy các bảng tần số, tỷ lệ % hoặc p-value của anh từ Tab 2 vào để viết bàn luận:", 
        placeholder="VD: Nhập 'Tỷ lệ nam/nữ là 1.42:1' hoặc dán nguyên cái bảng Crosstabs vào đây...",
        height=150
    )
    
    st.subheader("📝 Lệnh viết nhanh cho luận văn (Bấm là chạy):")
    
    citation_rules = """  
        QUY TẮC TRÍCH DẪN & HÀN LÂM BẮT BUỘC:
        1. BẮT BUỘC sử dụng kiểu trích dẫn số trong ngoặc vuông (VD: [1], [2]).
        2. TUYỆT ĐỐI KHÔNG dùng kiểu [Tên tác giả, Năm] (ví dụ: không dùng [Gordis, 2014]).
        3. Các số trích dẫn phải theo thứ tự xuất hiện liên tục trong bài.
        4. Cuối văn bản, BẮT BUỘC liệt kê danh mục 'Tài liệu tham khảo' tương ứng với các số đã dùng theo định dạng Vancouver.
        """
    # Chia làm 6 cột để hiển thị nút bấm
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # TẠO MÀN HÌNH HIỂN THỊ KẾT QUẢ FULL TRANG BÊN DƯỚI NÚT BẤM
    st.write("---")
    ket_qua_container = st.container()
    
    with col1:
        if st.button("Viết Đặt vấn đề"):
            with st.spinner("AI đang viết..."):
                prompt = f"Dựa trên Ngân hàng y văn, viết 'Đặt vấn đề' luận văn CKI Dược lâm sàng. Không dùng Heading 1 (#) để tránh chữ quá to, chỉ dùng Heading 3 (###). {citation_rules}"
                full_prop = f"NGÂN HÀNG Y VĂN:\n{st.session_state['ngan_hang_y_van']}\n\nYêu cầu: {prompt}"
                response = model.generate_content(full_prop, generation_config=generation_config)
                with ket_qua_container:
                    st.markdown(response.text)
                
    with col2:
        if st.button("Viết Tổng quan"):
            with st.spinner("AI đang viết..."):
                prompt = f"Viết phần tổng quan y văn chuyên sâu, tổng hợp các kết quả từ Ngân hàng y văn. Không dùng Heading 1 (#). {citation_rules}"
                full_prop = f"NGÂN HÀNG Y VĂN:\n{st.session_state['ngan_hang_y_van']}\n\nYêu cầu: {prompt}"
                response = model.generate_content(full_prop, generation_config=generation_config)
                with ket_qua_container:
                    st.markdown(response.text)
                
    with col3:
        if st.button("Phương pháp NC"):
            with st.spinner("AI đang thiết kế Chương 2..."):
                prompt = f"""
                Viết "Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU". Không dùng Heading 1 (#).
                Mục 2.2.3 BẮT BUỘC kẻ Bảng Markdown 5 cột (TT | Tên chỉ tiêu | Định nghĩa | Phân loại | Kỹ thuật thu thập). 
                {citation_rules}
                """
                full_prop = f"NGÂN HÀNG Y VĂN:\n{st.session_state['ngan_hang_y_van']}\n\nYêu cầu: {prompt}"
                response = model.generate_content(full_prop, generation_config=generation_config)
                with ket_qua_container:
                    st.markdown(response.text)
                
    with col4:
        if st.button("Viết Bàn luận toàn diện"):
            if not my_research_data:
                st.warning("Anh cần nhập số liệu của mình vào ô 'Bộ nhớ Số liệu' trước!")
            else:
                with st.spinner("AI đang viết Bàn luận phân đoạn theo tiêu đề chuẩn..."):
                    prompt = f"""
                    KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:
                    {my_research_data}
                    
                    YÊU CẦU TRÌNH BÀY (BẮT BUỘC TUÂN THỦ):
                    1. CHỈ SỬ DỤNG TRÍCH DẪN SỐ [1], [2]. TUYỆT ĐỐI KHÔNG DÙNG [Tên, Năm].
                    2. CHIA BÀN LUẬN THÀNH CÁC TIÊU ĐỀ PHỤ (TIỂU MỤC) CHUẨN XÁC DỰA TRÊN SỐ LIỆU ĐÃ CUNG CẤP. Sử dụng định dạng Heading 3 (ví dụ: ### 4.1. Đặc điểm bệnh nhân và phẫu thuật, ### 4.2. Thực trạng sử dụng kháng sinh, ### 4.3. Kết quả điều trị và so sánh...). Tuyệt đối không dùng Heading 1 hoặc 2.
                    3. Dưới mỗi tiêu đề, BẮT BUỘC viết thành các ĐOẠN VĂN HOÀN CHỈNH, mạch lạc, liên tục. TUYỆT ĐỐI KHÔNG dùng dạng liệt kê gạch đầu dòng (bullet points) để mô tả số liệu.
                    4. TRONG MỖI ĐOẠN VĂN BÀN LUẬN, phải kết hợp nhịp nhàng theo đúng cấu trúc:
                       - Nêu số liệu thực tế của tôi.
                       - Bàn luận và giải thích nguyên nhân y khoa (cơ chế, đặc thù tại viện).
                       - Lồng ghép so sánh, đối chiếu trực tiếp (cao hơn, thấp hơn, tương đồng) với số liệu của các tác giả trong NGÂN HÀNG Y VĂN ngay trong cùng đoạn văn đó.
                    5. Văn phong chuyên khảo y khoa hàn lâm, logic, không dùng từ ngữ cảm xúc.
                    {citation_rules}
                    """
                    full_prop = f"NGÂN HÀNG Y VĂN:\n{st.session_state['ngan_hang_y_van']}\n\nYêu cầu: {prompt}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
                    with ket_qua_container:
                        st.markdown(response.text) 
                        
    with col5:
        if st.button("So sánh NC liên quan"):
            if not my_research_data:
                st.warning("Anh cần nhập số liệu của mình vào ô 'Bộ nhớ Số liệu' trước!")
            else:
                with st.spinner("AI đang đối chiếu Y văn và viết phần 4.2, 4.3, 4.4..."):
                    prompt = f"""
                    KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:
                    {my_research_data}
                    
                    YÊU CẦU ĐẦU RA BẮT BUỘC (Trình bày đúng các cấu trúc tiểu mục sau, không dùng Heading 1 hoặc 2):
                    ### 4.2.2. So sánh với các nghiên cứu và khuyến cáo
                    (Lấy số liệu của TÔI làm gốc. Trích xuất thông tin từ NGÂN HÀNG Y VĂN để đối chiếu trực tiếp (cao hơn, thấp hơn, hay tương đồng). BẮT BUỘC giải thích sâu sắc nguyên nhân của sự khác biệt dựa trên: cỡ mẫu, đặc thù kỹ thuật, phương pháp, sự tuân thủ khuyến cáo).
                    
                    ### 4.3. Ý nghĩa lâm sàng và thực tiễn
                    (Rút ra bài học từ nghiên cứu này. Đề xuất các thay đổi thực tiễn để tối ưu hóa quy trình, giảm chi phí, nâng cao hiệu quả điều trị).
                    
                    ### 4.4. Hạn chế của nghiên cứu
                    (Tự đưa ra 2-3 hạn chế logic về cỡ mẫu, thời gian, phương pháp hồi cứu...).
                    
                    {citation_rules}
                    """
                    full_prop = f"NGÂN HÀNG Y VĂN:\n{st.session_state['ngan_hang_y_van']}\n\nYêu cầu: {prompt}"
                    response = model.generate_content(full_prop, generation_config=generation_config)
                    with ket_qua_container:
                        st.markdown(response.text)
                
    with col6:
        if st.button("Trích dẫn TLTK"):
            with st.spinner("AI đang lập danh mục..."):
                prompt = f"Lập danh mục tài liệu tham khảo Vancouver từ các tác giả trong Ngân hàng y văn. Không dùng Heading 1 (#)."
                full_prop = f"NGÂN HÀNG Y VĂN:\n{st.session_state['ngan_hang_y_van']}\n\nYêu cầu: {prompt}"
                response = model.generate_content(full_prop, generation_config=generation_config)
                with ket_qua_container:
                    st.markdown(response.text)
    
    st.write("---")
    custom_prompt = st.text_area("Nhập câu lệnh khác ở đây:")
    if st.button("Chạy lệnh"):
        if custom_prompt:
            with st.spinner("AI đang xử lý..."):
                anti_hallucination = "\nLƯU Ý NGHIÊM NGẶT: Không tự bịa thông tin. Trích dẫn số liệu cụ thể."
                full_prop = f"NGÂN HÀNG Y VĂN:\n{st.session_state['ngan_hang_y_van']}\n\nYêu cầu: {custom_prompt}\n{anti_hallucination}\n{citation_rules}"
                response = model.generate_content(full_prop, generation_config=generation_config)
                st.markdown(response.text)
        else:
            st.warning("Vui lòng nhập yêu cầu!")                
        with st.expander("📂 Xem danh sách các file PDF đã tải lên"):
            for f in uploaded_files:
                st.text(f"- {f.name} ({round(f.size / 1024, 1)} KB)")
# ----------------------------------------------------
# TAB 2: PHÂN TÍCH SỐ LIỆU TỪ EXCEL (MÔ PHỎNG SPSS)
# ----------------------------------------------------
with tab2:
    st.header("📊 Phân tích thống kê & Kiểm định")
    
    excel_file = st.file_uploader("Tải lên file số liệu bệnh án (Excel .xlsx)", type=["xlsx", "xls"], key="excel_uploader")
    
    if excel_file is not None:
        df = pd.read_excel(excel_file)
        columns = df.columns.tolist()
        
        st.success(f"Đọc dữ liệu thành công! File có {df.shape[0]} dòng và {df.shape[1]} cột.")
        
        with st.expander("👀 Xem trước dữ liệu gốc"):
            st.dataframe(df.head())

        st.write("---")
        st.subheader("🛠️ CÔNG CỤ PHÂN TÍCH CHUYÊN SÂU")
        
        # --- TÍNH NĂNG 1: THỐNG KÊ TẦN SỐ ---
        st.markdown("### 1. Thống kê mô tả")
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
        
        st.markdown("### 2. Bảng chéo & Phân tích mối liên quan")
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
        
        st.markdown("### 3. Phân tích thêm số liệu khác nếu cần")
        analysis_prompt = st.text_area("Nhập yêu cầu:")
        
        if st.button("Chạy lệnh"):
            if analysis_prompt:
                with st.spinner("AI đang xử lý..."):
                    desc_stats = df.describe(include='all').to_string()
                    data_prompt = f"Bảng thống kê tổng quát:\n{desc_stats}\n\nYêu cầu: {analysis_prompt}. Viết văn phong hàn lâm, khô khan."
                    response = model.generate_content(data_prompt, generation_config=generation_config)
                    st.markdown(response.text)
            else:
                st.warning("Vui lòng nhập yêu cầu!")
