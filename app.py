import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import time
from google.api_core.exceptions import ResourceExhausted

# THƯ VIỆN KIẾN TRÚC RAG (CHUẨN MỚI NHẤT, KHÔNG LỖI)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="NCKH - Hỗ trợ Nghiên cứu", layout="wide")

custom_css = """
<style>
    /* Ép phông chữ Arial an toàn */
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li { font-family: 'Arial', sans-serif; }
    
    /* Nền toàn bộ trang web (Gradient xanh dương - tím nhạt) */
    .stApp { background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); }
    
    /* Chỉnh màu và hiệu ứng cho Tiêu đề chính */
    h1 { color: #4a148c !important; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); font-weight: 800; margin-bottom: 30px; }
    
    /* Nền trắng mờ cho các Tab */
    .stTabs [data-baseweb="tab-panel"] { background-color: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
    
    /* Trang trí thanh Tab */
    .stTabs [data-baseweb="tab-list"] { background-color: rgba(255, 255, 255, 0.5); border-radius: 10px; padding: 5px; }
    
    /* Làm đẹp các nút bấm (Button) */
    div.stButton > button { background-color: #6a1b9a !important; color: white !important; font-weight: bold; border-radius: 8px; border: none; padding: 10px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; }
    div.stButton > button:hover { background-color: #4a148c !important; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.2); }
    
    /* Tùy chỉnh bảng dữ liệu Dataframe */
    [data-testid="stDataFrame"] { background-color: white; border-radius: 10px; padding: 10px; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.title("HỖ TRỢ NGHIÊN CỨU KHOA HỌC")

# ==========================================
# CẤU HÌNH API GEMINI & MÔ HÌNH NHÚNG
# ==========================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    system_prompt = """
    Bạn là một chuyên gia dược lâm sàng, thống kê y học và biên tập viên luận văn y khoa cấp cao. 
    Quy tắc làm việc của bạn:
    1. VĂN PHONG (QUAN TRỌNG NHẤT): Tuyệt đối KHÔNG sử dụng từ ngữ hoa mỹ, bay bổng, sáo rỗng. Văn phong phải cực kỳ khô khan, trực diện, logic chặt chẽ.
    2. TÍNH CHUYÊN SÂU: Sử dụng chính xác 100% thuật ngữ chuyên ngành. Tập trung vào bằng chứng (Evidence-based), cơ sở sinh lý bệnh, cơ chế dược lý, số liệu dịch tễ.
    3. CHỐNG BỊA ĐẶT TUYỆT ĐỐI (ANTI-HALLUCINATION): TUYỆT ĐỐI KHÔNG tự bịa số liệu. Chỉ sử dụng dữ liệu từ 'TÀI LIỆU Y VĂN TRÍCH XUẤT'. Nếu không có thông tin, BẮT BUỘC trả lời: 'Tài liệu không đề cập'.
    4. TRÍCH DẪN: Mỗi khẳng định bắt buộc kèm theo trích dẫn số [1], [2]. Tuyệt đối KHÔNG dùng [Tên tác giả, Năm].
    5. BẢNG BIỂU: Trình bày dưới dạng bảng Markdown chuẩn.
    """
    
    # SỬ DỤNG ĐÚNG TÊN MODEL CHUẨN CỦA GOOGLE
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    
    # HÀM BỌC AN TOÀN: Tự động né lỗi quá tải API
    def safe_generate_content(prompt, config=generation_config, max_retries=10):
        for attempt in range(max_retries):
            try:
                return model.generate_content(prompt, generation_config=config)
            except ResourceExhausted:
                if attempt < max_retries - 1:
                    wait_time = 15 
                    status_msg = st.warning(f"⏳ Trạm máy chủ Google đang bận. Ứng dụng tự động nghỉ {wait_time} giây và chạy lại (Lần thử {attempt + 2}/10)...")
                    time.sleep(wait_time)
                    status_msg.empty() 
                else:
                    st.info("⏱️ Dữ liệu đầu vào đợt này quá lớn nên máy chủ chưa kịp xử lý. Anh vui lòng tải lại trang (F5) và chia nhỏ số file ra nhé!")
                    return None
            except Exception as e:
                st.warning(f"⚠️ Có gián đoạn kết nối: {e}")
                return None

except Exception as e:
    st.error("Chưa cấu hình API Key trong Streamlit Secrets. Vui lòng thêm khóa vào mục cài đặt của ứng dụng.")
    st.stop()

# Cache Mô hình nhúng HuggingFace (Miễn phí, Siêu nhẹ)
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    embeddings = load_embedding_model()
except Exception as e:
    st.error(f"Lỗi tải mô hình nhúng: {e}")
    st.stop()

# ==========================================
# CÁC TAB CHỨC NĂNG
# ==========================================
tab1, tab2 = st.tabs(["📄 Đọc Tài liệu & Viết Luận văn (RAG)", "📊 Phân tích Số liệu Bệnh án (Excel)"])

# ----------------------------------------------------
# TAB 1: RAG VECTOR DATABASE
# ----------------------------------------------------
with tab1:
    st.header("Phân tích tài liệu và Viết bài")
    
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
        
    st.markdown("### 🏦 Ngân hàng Y văn (Cơ sở dữ liệu Vector SKLearn)")
    st.info("💡 Tải file PDF lên và bấm 'Xử lý & Mã hóa Vector'. Hệ thống sẽ tự động băm nhỏ tài liệu để tìm kiếm siêu tốc, chống quá tải API.")
    
    uploaded_files = st.file_uploader("Tải lên tài liệu nghiên cứu (PDF) để AI đọc:", type="pdf", accept_multiple_files=True, key="pdf_uploader")
    
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
                    # Chia nhỏ tài liệu thành các đoạn 1500 ký tự
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                    chunks = text_splitter.split_text(combined_text)
                    
                    # Tạo nhúng với SKLearn (Rất nhẹ, chạy hoàn toàn bằng CPU)
                    if st.session_state["vector_store"] is None:
                        st.session_state["vector_store"] = SKLearnVectorStore.from_texts(chunks, embedding=embeddings)
                    else:
                        st.session_state["vector_store"].add_texts(chunks)
                        
                    st.success(f"✅ Đã lập chỉ mục Vector thành công {len(chunks)} phân đoạn dữ liệu. RAG đã sẵn sàng hoạt động!")
                else:
                    st.error("Lỗi: Không đọc được chữ từ file PDF này (có thể là file ảnh chụp).")
                
    st.write("---")
    
    st.markdown("### 🌉 Bộ nhớ Số liệu của riêng bạn (Dành cho phần Bàn luận)")
    my_research_data = st.text_area(
        "Copy các bảng tần số, tỷ lệ % hoặc p-value của anh từ Tab 2 vào để viết bàn luận:", 
        placeholder="VD: Nhập 'Tỷ lệ nam/nữ là 1.42:1' hoặc dán nguyên cái bảng Crosstabs vào đây...",
        height=150
    )
    
    st.subheader("📝 Lệnh viết nhanh cho luận văn (Bấm là chạy):")
    citation_rules = "BẮT BUỘC sử dụng trích dẫn số [1], [2]. Tuyệt đối KHÔNG dùng [Tên tác giả, Năm]. Liệt kê TLTK chuẩn Vancouver ở cuối."
        
    col1, col2, col3, col4 = st.columns(4)
    st.write("---")
    ket_qua_container = st.container()
    
    # HÀM TRUY XUẤT RAG TỰ ĐỘNG
    def retrieve_context(query, k=6):
        if st.session_state["vector_store"] is not None:
            docs = st.session_state["vector_store"].similarity_search(query, k=k)
            return "\n\n".join([f"--- Đoạn trích ---:\n{d.page_content}" for d in docs])
        return "Không có dữ liệu y văn trong Vector Database."

    with col1:
        if st.button("Viết Đặt vấn đề", key="btn_dt_tq"):
            with st.spinner("AI đang quét Vector Database và viết..."):
                query = "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, tổng quan dịch tễ học"
                context = retrieve_context(query)
                
                prompt = f"Dựa trên các đoạn trích y văn, viết Đặt vấn đề và Tổng quan. Chỉ dùng Heading 3 (###). {citation_rules}"
                full_prop = f"TÀI LIỆU Y VĂN TRÍCH XUẤT TỪ VECTOR DB:\n{context}\n\nYêu cầu: {prompt}"
                
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container: st.markdown(response.text)
                
    with col2:
        if st.button("Phương pháp NC", key="btn_pp"):
            with st.spinner("AI đang thiết kế Chương 2..."):
                query = "Đối tượng nghiên cứu, tiêu chuẩn nhận loại trừ, thiết kế nghiên cứu, cỡ mẫu"
                context = retrieve_context(query, k=4)
                
                prompt = f"""Viết "Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU". Chỉ dùng Heading 3. 
                BẮT BUỘC kẻ Bảng Markdown 5 cột (TT | Tên chỉ tiêu | Định nghĩa | Phân loại | Kỹ thuật thu thập). {citation_rules}"""
                full_prop = f"TÀI LIỆU Y VĂN TRÍCH XUẤT TỪ VECTOR DB:\n{context}\n\nYêu cầu: {prompt}"
                
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container: st.markdown(response.text)
                
    with col3:
        if st.button("Viết Bàn luận (So sánh)", key="btn_bl"):
            if not my_research_data:
                st.warning("Anh cần nhập số liệu của mình vào ô 'Bộ nhớ Số liệu' trước!")
            else:
                with st.spinner("AI đang tìm kiếm các đoạn y văn có tương đồng với số liệu của anh..."):
                    context = retrieve_context(my_research_data, k=8)
                    
                    prompt = f"""
                    KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:
                    {my_research_data}
                    
                    YÊU CẦU TRÌNH BÀY (BẮT BUỘC TUÂN THỦ):
                    1. Chia bàn luận thành các tiêu đề phụ (Heading 3). KHÔNG gạch đầu dòng liệt kê.
                    2. TRONG MỖI ĐOẠN VĂN: Nêu số liệu thực tế của tôi -> Bàn luận giải thích nguyên nhân y khoa -> Lồng ghép SO SÁNH trực tiếp với số liệu trong TÀI LIỆU Y VĂN TRÍCH XUẤT.
                    3. Văn phong chuyên khảo y khoa hàn lâm. {citation_rules}
                    """
                    full_prop = f"TÀI LIỆU Y VĂN TRÍCH XUẤT TỪ VECTOR DB (Chỉ dùng dữ liệu này để so sánh):\n{context}\n\nYêu cầu: {prompt}"
                    
                    response = safe_generate_content(full_prop)
                    if response:
                        with ket_qua_container: st.markdown(response.text) 
                            
    with col4:
        if st.button("Trích dẫn TLTK", key="btn_tltk"):
            with st.spinner("AI đang lập danh mục..."):
                query = "Tài liệu tham khảo, References, Tên tác giả, Năm xuất bản"
                context = retrieve_context(query, k=10)
                
                prompt = f"Trích xuất tên các tác giả và bài báo, lập danh mục Tài liệu tham khảo chuẩn Vancouver."
                full_prop = f"TÀI LIỆU Y VĂN TRÍCH XUẤT:\n{context}\n\nYêu cầu: {prompt}"
                
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container: st.markdown(response.text)
    
    st.write("---")
    custom_prompt = st.text_area("Hỏi đáp trực tiếp với kho tài liệu (Chat with PDF):", key="custom_prompt_tab1")
    if st.button("Chạy lệnh", key="btn_custom_tab1"):
        if custom_prompt:
            with st.spinner("Đang lục tìm trong Database..."):
                context = retrieve_context(custom_prompt, k=5)
                full_prop = f"TÀI LIỆU TRÍCH XUẤT:\n{context}\n\nCâu hỏi: {custom_prompt}\n(Lưu ý: Chỉ trả lời dựa trên tài liệu trích xuất, tuyệt đối không bịa. {citation_rules})"
                response = safe_generate_content(full_prop)
                if response: st.markdown(response.text)
        else:
            st.warning("Vui lòng nhập yêu cầu!")
            
    with st.expander("📂 Xem danh sách các file PDF đã tải lên"):
        if uploaded_files:
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
        
        # --- THỐNG KÊ MÔ TẢ ---
        st.markdown("### 1. Thống kê mô tả")
        st.info("💡 Mẹo: Chọn nhiều biến cùng lúc để AI tự động đếm tần số và nhận xét hàng loạt. KHÔNG chọn biến định danh (Họ tên, Số bệnh án...).")
        
        vars_desc = st.multiselect("🏷️ Chọn TẤT CẢ các biến cần thống kê (VD: Giới tính, Nhóm tuổi...):", columns, key="auto_var_desc")
        
        if st.button("🚀 Chạy toàn bộ Thống kê & Nhận xét", key="btn_stat_desc"):
            if not vars_desc:
                st.warning("Vui lòng chọn ít nhất 1 biến để phân tích!")
            else:
                for var in vars_desc:
                    with st.spinner(f"AI đang tính toán tần số cho biến '{var}'..."):
                        freq_table = df[var].value_counts().to_string()
                        total = len(df[var].dropna())
                        
                        prompt = f"""
                        Dữ liệu đếm thực tế của biến '{var}': {freq_table} (Tổng: {total}).
                        Yêu cầu: 1. Vẽ bảng SPSS (Phân loại, n, %). 2. Viết nhận xét y khoa chuyên sâu, khô khan.
                        """
                        response = safe_generate_content(prompt)
                        
                        if response:
                            st.subheader(f"► Phân tích biến: {var}")
                            st.markdown(response.text)
                            st.write("---")
                        time.sleep(3)
        st.write("---")
        
        # --- BẢNG CHÉO ---
        st.markdown("### 2. Bảng chéo & Phân tích mối liên quan")
        st.info("💡 Mẹo: Chọn 1 Biến phụ thuộc (Cột) làm gốc. Sau đó chọn nhiều Biến độc lập (Hàng).")
        
        target_col = st.selectbox("🎯 Chọn Biến Phụ thuộc / Cột (VD: Kết quả điều trị):", columns, key="auto_target")
        indep_cols = st.multiselect("🏷️ Chọn TẤT CẢ các Biến Độc lập / Hàng:", columns, key="auto_indep")
            
        if st.button("🚀 Chạy toàn bộ Bảng chéo & Nhận xét", key="btn_crosstab"):
            if not indep_cols:
                st.warning("Vui lòng chọn ít nhất 1 biến độc lập (hàng) ở ô phía trên!")
            else:
                for var in indep_cols:
                    if var == target_col:
                        continue
                        
                    with st.spinner(f"AI đang xử lý bảng chéo giữa '{var}' và '{target_col}'..."):
                        crosstab_df = pd.crosstab(df[var], df[target_col])
                        
                        prompt = f"""
                        Bảng Crosstabs thực tế giữa '{var}' và '{target_col}':\n{crosstab_df.to_string()}\n
                        Yêu cầu: 1. Trình bày bảng khoa học (% hàng/cột). 2. Nhận xét chuyên sâu.
                        """
                        response = safe_generate_content(prompt)
                        
                        if response:
                            st.subheader(f"► Mối liên quan giữa {var} và {target_col}")
                            st.markdown(response.text)
                            st.write("---")
                        time.sleep(3)
        st.write("---")
        
        # --- PHÂN TÍCH KHÁC ---
        st.markdown("### 3. Phân tích thêm số liệu khác nếu cần")
        analysis_prompt = st.text_area("Nhập yêu cầu:", key="analysis_prompt_tab2")
        
        if st.button("Chạy lệnh xử lý số liệu", key="btn_custom_tab2"):
            if analysis_prompt:
                with st.spinner("AI đang xử lý..."):
                    desc_stats = df.describe(include='all').to_string()
                    data_prompt = f"Bảng thống kê tổng quát:\n{desc_stats}\n\nYêu cầu: {analysis_prompt}. Viết văn phong hàn lâm."
                    response = safe_generate_content(data_prompt)
                    if response:
                        st.markdown(response.text)
            else:
                st.warning("Vui lòng nhập yêu cầu!")
