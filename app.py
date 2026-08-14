import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import time
from google.api_core.exceptions import ResourceExhausted

# THƯ VIỆN MỚI CHO KIẾN TRÚC RAG (VECTOR DATABASE)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="NCKH - Hệ thống RAG", layout="wide")

custom_css = """
<style>
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li { font-family: 'Arial', sans-serif; }
    .stApp { background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); }
    h1 { color: #4a148c !important; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); font-weight: 800; margin-bottom: 30px; }
    .stTabs [data-baseweb="tab-panel"] { background-color: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
    .stTabs [data-baseweb="tab-list"] { background-color: rgba(255, 255, 255, 0.5); border-radius: 10px; padding: 5px; }
    div.stButton > button { background-color: #6a1b9a !important; color: white !important; font-weight: bold; border-radius: 8px; border: none; padding: 10px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; }
    div.stButton > button:hover { background-color: #4a148c !important; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.2); }
    [data-testid="stDataFrame"] { background-color: white; border-radius: 10px; padding: 10px; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.title("HỖ TRỢ NGHIÊN CỨU KHOA HỌC")

# ==========================================
# CẤU HÌNH API GEMINI & RAG EMBEDDINGS
# ==========================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    # KHỞI TẠO MÔ HÌNH NHÚNG (Biến văn bản thành ma trận số Vector)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    system_prompt = """
    Bạn là một chuyên gia thống kê y học và biên tập viên luận văn y khoa cấp cao. 
    Quy tắc:
    1. TUYỆT ĐỐI KHÔNG sử dụng từ ngữ hoa mỹ. Văn phong cực kỳ khô khan, trực diện.
    2. CHỐNG BỊA ĐẶT (ANTI-HALLUCINATION): Chỉ sử dụng dữ liệu từ 'TÀI LIỆU Y VĂN TRÍCH XUẤT' để đối chiếu. Nếu không có thông tin, trả lời: 'Tài liệu không đề cập'.
    3. Trích dẫn số dạng [1], [2]. Tuyệt đối KHÔNG dùng [Tên tác giả, Năm].
    """
    
    model = genai.GenerativeModel("gemini-3.6-flash", system_instruction=system_prompt)
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    
    def safe_generate_content(prompt, config=generation_config, max_retries=10):
        for attempt in range(max_retries):
            try:
                return model.generate_content(prompt, generation_config=config)
            except ResourceExhausted:
                if attempt < max_retries - 1:
                    wait_time = 15
                    status_msg = st.warning(f"⏳ Băng thông đang nghẽn. Tự động nghỉ {wait_time}s (Thử lại lần {attempt + 2}/10)...")
                    time.sleep(wait_time)
                    status_msg.empty()
                else:
                    st.info("⏱️ API quá tải. Vui lòng thử lại sau vài phút!")
                    return None
            except Exception as e:
                st.warning(f"⚠️ Có lỗi kết nối: {e}")
                return None

except Exception as e:
    st.error("Chưa cấu hình API Key trong Streamlit Secrets.")
    st.stop()

# ==========================================
# CÁC TAB CHỨC NĂNG
# ==========================================
tab1, tab2 = st.tabs(["📄 Đọc Tài liệu & Viết Luận văn (RAG)", "📊 Phân tích Số liệu Bệnh án (Excel)"])

# ----------------------------------------------------
# TAB 1: KIẾN TRÚC RAG - TÌM KIẾM VECTOR
# ----------------------------------------------------
with tab1:
    st.header("Trí tuệ nhân tạo RAG - Đọc & Phân tích y văn")
    
    # Khởi tạo Vector Store trong bộ nhớ tạm
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
        
    st.markdown("### 🏦 Ngân hàng Y văn (Cơ sở dữ liệu Vector)")
    st.info("💡 Tải file PDF lên và bấm 'Mã hóa Vector'. Ứng dụng sẽ băm nhỏ hàng ngàn trang tài liệu thành các không gian toán học để tìm kiếm siêu tốc, triệt tiêu hoàn toàn lỗi quá tải API.")
    
    uploaded_files = st.file_uploader("Tải lên tài liệu nghiên cứu (PDF):", type="pdf", accept_multiple_files=True, key="pdf_uploader")
    
    if uploaded_files:
        if st.button("📥 Xử lý & Mã hóa Vector Database", type="primary"):
            with st.spinner("AI đang băm nhỏ tài liệu và chuyển đổi thành Vector..."):
                combined_text = ""
                for index, uploaded_file in enumerate(uploaded_files, start=1):
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            combined_text += page_text + "\n"
                
                if combined_text.strip():
                    # 1. BĂM NHỎ TÀI LIỆU (Chunking) - Đoạn 1500 ký tự, gối nhau 300
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                    chunks = text_splitter.split_text(combined_text)
                    
                    # 2. TẠO NHÚNG (Embedding) & LƯU VÀO FAISS
                    new_vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    
                    if st.session_state["vector_store"] is None:
                        st.session_state["vector_store"] = new_vector_store
                    else:
                        st.session_state["vector_store"].merge_from(new_vector_store)
                        
                    st.success(f"✅ Đã lập chỉ mục Vector thành công {len(chunks)} phân đoạn dữ liệu. RAG đã sẵn sàng hoạt động!")
                else:
                    st.error("Lỗi: Không đọc được chữ từ file PDF này (có thể là file ảnh chụp).")
                
    st.write("---")
    
    st.markdown("### 🌉 Bộ nhớ Số liệu của riêng bạn (Dành cho phần Bàn luận)")
    my_research_data = st.text_area("Copy bảng tần số, tỷ lệ % hoặc p-value của anh từ Tab 2 vào đây:", height=150)
    
    st.subheader("📝 Lệnh viết nhanh cho luận văn (RAG Retrieval):")
    citation_rules = "BẮT BUỘC sử dụng trích dẫn số [1], [2]. Tuyệt đối KHÔNG dùng [Tên tác giả, Năm]. Liệt kê TLTK chuẩn Vancouver ở cuối."
        
    col1, col2, col3, col4 = st.columns(4)
    st.write("---")
    ket_qua_container = st.container()
    
    # HÀM TRUY XUẤT RAG (RETRIEVAL)
    def retrieve_context(query, k=5):
        if st.session_state["vector_store"] is not None:
            docs = st.session_state["vector_store"].similarity_search(query, k=k)
            return "\n\n".join([f"--- Đoạn trích ---:\n{d.page_content}" for d in docs])
        return "Không có dữ liệu y văn trong Vector Database."

    with col1:
        if st.button("Viết Đặt vấn đề & Tổng quan", key="btn_dt_tq"):
            with st.spinner("AI đang quét Vector Database và viết..."):
                query = "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, tổng quan dịch tễ học"
                context = retrieve_context(query, k=6)
                
                prompt = f"Dựa trên các đoạn trích y văn, viết Đặt vấn đề và Tổng quan. Sử dụng Heading 3 (###). {citation_rules}"
                full_prop = f"TÀI LIỆU Y VĂN TRÍCH XUẤT TỪ VECTOR DB:\n{context}\n\nYêu cầu: {prompt}"
                
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container: st.markdown(response.text)
                
    with col2:
        if st.button("Phương pháp NC", key="btn_pp"):
            with st.spinner("AI đang quét Vector Database..."):
                query = "Đối tượng nghiên cứu, tiêu chuẩn nhận loại trừ, thiết kế nghiên cứu, cỡ mẫu"
                context = retrieve_context(query, k=4)
                
                prompt = f"""Viết "Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP". Sử dụng Heading 3. 
                BẮT BUỘC kẻ Bảng Markdown 5 cột (TT | Tên chỉ tiêu | Định nghĩa | Phân loại | Kỹ thuật thu thập). {citation_rules}"""
                full_prop = f"TÀI LIỆU Y VĂN TRÍCH XUẤT TỪ VECTOR DB:\n{context}\n\nYêu cầu: {prompt}"
                
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container: st.markdown(response.text)
                
    with col3:
        if st.button("Viết Bàn luận (So sánh Y văn)", key="btn_bl"):
            if not my_research_data:
                st.warning("Anh cần nhập số liệu của mình vào ô 'Bộ nhớ Số liệu' trước!")
            else:
                with st.spinner("AI đang tìm kiếm các đoạn y văn có tương đồng với số liệu của anh..."):
                    # KỲ DIỆU CỦA RAG: Dùng chính số liệu của người dùng làm từ khóa tìm kiếm
                    context = retrieve_context(my_research_data, k=8)
                    
                    prompt = f"""
                    KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:
                    {my_research_data}
                    
                    YÊU CẦU TRÌNH BÀY:
                    1. Chia bàn luận thành các tiểu mục (Heading 3). KHÔNG gạch đầu dòng liệt kê.
                    2. TRONG MỖI ĐOẠN VĂN: Nêu số liệu của tôi -> Giải thích cơ chế -> SO SÁNH trực tiếp với số liệu trong TÀI LIỆU Y VĂN TRÍCH XUẤT.
                    3. Văn phong hàn lâm, logic. {citation_rules}
                    """
                    full_prop = f"TÀI LIỆU Y VĂN TRÍCH XUẤT TỪ VECTOR DB (Chỉ dùng dữ liệu này để so sánh):\n{context}\n\nYêu cầu: {prompt}"
                    
                    response = safe_generate_content(full_prop)
                    if response:
                        with ket_qua_container: st.markdown(response.text) 
                            
    with col4:
        if st.button("Lập danh mục TLTK", key="btn_tltk"):
            with st.spinner("AI đang lập danh mục..."):
                query = "Tài liệu tham khảo, References, Tên tác giả, Năm xuất bản"
                context = retrieve_context(query, k=10)
                
                prompt = f"Trích xuất tên các tác giả và bài báo trong văn bản, lập danh mục Tài liệu tham khảo chuẩn Vancouver."
                full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {prompt}"
                
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container: st.markdown(response.text)
    
    st.write("---")
    custom_prompt = st.text_area("Hỏi đáp trực tiếp với kho tài liệu (Chat with PDF):", key="custom_prompt_tab1")
    if st.button("Hỏi AI", key="btn_custom_tab1"):
        if custom_prompt:
            with st.spinner("Đang lục tìm trong Database..."):
                context = retrieve_context(custom_prompt, k=5)
                full_prop = f"TÀI LIỆU TRÍCH XUẤT:\n{context}\n\nCâu hỏi: {custom_prompt}\n(Chỉ trả lời dựa trên tài liệu trích xuất, không tự bịa)"
                response = safe_generate_content(full_prop)
                if response: st.markdown(response.text)
        else:
            st.warning("Vui lòng nhập câu hỏi!")
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
        
        if st.button("Chạy lệnh xử lý số liệu"):
            if analysis_prompt:
                with st.spinner("AI đang xử lý..."):
                    desc_stats = df.describe(include='all').to_string()
                    data_prompt = f"Bảng thống kê tổng quát:\n{desc_stats}\n\nYêu cầu: {analysis_prompt}. Viết văn phong hàn lâm, khô khan."
                    response = model.generate_content(data_prompt, generation_config=generation_config)
                    st.markdown(response.text)
            else:
                st.warning("Vui lòng nhập yêu cầu!")
