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
    /* ===== FONT & TỔNG THỂ ===== */
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700;800&display=swap');
    
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li, .stMarkdown {
        font-family: 'Be Vietnam Pro', 'Arial', sans-serif;
    }
    
    /* ===== NỀN TOÀN TRANG - GRADIENT ĐA SẮC ĐỘNG ===== */
    .stApp {
        background: linear-gradient(-45deg, #ff9a9e, #a18cd1, #667eea, #43e97b, #38f9d7);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* ===== TIÊU ĐỀ CHÍNH ===== */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 800;
        font-size: 2.6rem !important;
        letter-spacing: 1px;
        margin-bottom: 30px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.35), 0 0 30px rgba(255,255,255,0.25);
        -webkit-text-stroke: 0.5px rgba(255,255,255,0.15);
    }
    
    h2, h3 {
        color: #4a148c !important;
        font-weight: 700;
    }
    
    /* ===== KHỐI NỘI DUNG TAB - HIỆU ỨNG KÍNH MỜ (GLASSMORPHISM) ===== */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
        border: 1px solid rgba(255,255,255,0.4);
    }
    
    /* ===== THANH TAB ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 6px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 700;
        color: #4a148c;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6a1b9a, #ab47bc) !important;
        color: #fff !important;
        box-shadow: 0 4px 10px rgba(106,27,154,0.4);
    }
    
    /* ===== NÚT BẤM - GRADIENT SẶC SỠ + HIỆU ỨNG HOVER ===== */
    div.stButton > button {
        background: linear-gradient(135deg, #6a1b9a 0%, #ab47bc 50%, #ff6ec4 100%) !important;
        color: white !important;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 22px;
        box-shadow: 0 6px 14px rgba(106,27,154,0.35);
        transition: all 0.25s ease;
        letter-spacing: 0.3px;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 22px rgba(106,27,154,0.5);
        filter: brightness(1.08);
    }
    div.stButton > button:active {
        transform: translateY(0px) scale(0.98);
    }
    
    /* ===== NÚT LOẠI PRIMARY (nổi bật hơn) ===== */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ff512f, #f09819) !important;
        box-shadow: 0 6px 14px rgba(255,81,47,0.4);
    }
    
    /* ===== BẢNG DỮ LIỆU ===== */
    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    /* ===== Ô NHẬP LIỆU (text_area, selectbox, multiselect...) ===== */
    .stTextArea textarea, .stTextInput input {
        border-radius: 12px !important;
        border: 1.5px solid #d1a3f0 !important;
        background-color: rgba(255,255,255,0.9) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #6a1b9a !important;
        box-shadow: 0 0 0 3px rgba(106,27,154,0.15) !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1.5px solid #d1a3f0 !important;
    }
    
    /* ===== EXPANDER / INFO BOX ===== */
    .streamlit-expanderHeader {
        background: rgba(171, 71, 188, 0.12);
        border-radius: 10px;
        font-weight: 600;
        color: #4a148c;
    }
    .stAlert {
        border-radius: 12px !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
        background: rgba(255,255,255,0.6);
        padding: 10px;
    }
    
    /* ===== SUBHEADER (► Phân tích biến...) ===== */
    .stMarkdown h3 {
        border-left: 5px solid #ab47bc;
        padding-left: 12px;
        margin-top: 10px;
    }
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
    model = genai.GenerativeModel("gemini-3.6-flash", system_instruction=system_prompt)
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
    if "ngan_hang_y_van" not in st.session_state:
        st.session_state["ngan_hang_y_van"] = ""
        
    st.markdown("### 🏦 Ngân hàng Y văn (Cơ sở dữ liệu Vector SKLearn)")
    st.info("💡 Tải file PDF lên và bấm 'Xử lý & Mã hóa Vector'. Hệ thống sẽ tự động băm nhỏ tài liệu để tìm kiếm siêu tốc, chống quá tải API.")
    
    uploaded_files = st.file_uploader("Tải lên tài liệu nghiên cứu (PDF) để AI đọc:", type="pdf", accept_multiple_files=True, key="pdf_uploader")
    
    if uploaded_files:
        col_up1, col_up2 = st.columns(2)
        
        with col_up1:
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
        
        with col_up2:
            if st.button("📝 Rút trích số liệu & Ghi vào Ngân hàng", type="primary"):
                combined_text = ""
                for uploaded_file in uploaded_files:
                    try:
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                combined_text += page_text + "\n"
                    except Exception as e:
                        st.session_state["last_error"] = f"Lỗi đọc file {uploaded_file.name}: {e}"

                if not combined_text.strip():
                    st.session_state["last_error"] = "Không đọc được chữ từ các file PDF này (có thể là file ảnh chụp/scan không có lớp text)."
                else:
                    with st.spinner(f"AI đang đọc {len(combined_text)} ký tự từ {len(uploaded_files)} file PDF..."):
                        extract_prompt = """
                        Hãy đọc các tài liệu PDF trên và TÓM TẮT CÔ ĐẶC lại những thông tin sau:
                        1. Tên tác giả, năm nghiên cứu, tên bài báo.
                        2. Mục tiêu nghiên cứu và đối tượng nghiên cứu.
                        3. Các kết quả, số liệu quan trọng nhất (tỷ lệ %, p-value, OR, RR...).
                        4. Kết luận chính của tác giả.
                        Tuyệt đối không bịa số liệu. Trình bày dưới dạng gạch đầu dòng ngắn gọn.
                        """
                        # Gemini có giới hạn input - nếu quá dài, cắt bớt để tránh lỗi
                        text_to_send = combined_text[:120000]
                        full_prop = f"Tài liệu gốc:\n{text_to_send}\n\nYêu cầu: {extract_prompt}"

                        try:
                            response = model.generate_content(full_prop, generation_config=generation_config)
                            if response and response.text:
                                st.session_state["ngan_hang_y_van"] += f"\n\n{response.text}"
                                st.session_state["last_error"] = None
                                st.session_state["last_success"] = f"✅ Đã rút trích và ghi thêm dữ liệu ({len(response.text)} ký tự)."
                            else:
                                st.session_state["last_error"] = "AI trả về phản hồi rỗng (có thể do bộ lọc an toàn chặn nội dung)."
                        except Exception as e:
                            st.session_state["last_error"] = f"Lỗi gọi API Gemini: {e}"

                st.rerun()

        # Hiển thị lỗi/thành công đã lưu lại XUYÊN SUỐT qua rerun (không bị mất)
        if st.session_state.get("last_error"):
            st.error(st.session_state["last_error"])
        if st.session_state.get("last_success"):
            st.success(st.session_state["last_success"])
        
        # Ô hiển thị + cho phép chỉnh sửa Ngân hàng y văn đã rút trích
        st.text_area(
            "📚 Ngân hàng y văn đã rút trích (có thể sửa tay trước khi dùng):",
            value=st.session_state["ngan_hang_y_van"],
            height=200,
            key="ngan_hang_y_van_display"
        )
        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Xóa Ngân hàng y văn"):
                st.session_state["ngan_hang_y_van"] = ""
                st.rerun()
                
    st.write("---")
    
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
            with st.spinner("AI đang quét Vector DB + Ngân hàng y văn và viết..."):
                query = "Đặt vấn đề, tính cấp thiết, lý do nghiên cứu, tổng quan dịch tễ học"
                context = build_context(query, k=6)
                prompt = f"Dựa trên TÀI LIỆU Y VĂN được cung cấp, viết 'Đặt vấn đề' luận văn CKI Dược lâm sàng. Không dùng Heading 1 (#) để tránh chữ quá to, chỉ dùng Heading 3 (###). {citation_rules}"
                full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {prompt}"
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container:
                        st.markdown(response.text)
                
    with col2:
        if st.button("Viết Tổng quan"):
            with st.spinner("AI đang quét Vector DB + Ngân hàng y văn và viết..."):
                query = "Tổng quan y văn, các nghiên cứu liên quan, kết quả chính, kết luận"
                context = build_context(query, k=8)
                prompt = f"Viết phần tổng quan y văn chuyên sâu, tổng hợp các kết quả từ TÀI LIỆU Y VĂN được cung cấp. Không dùng Heading 1 (#). {citation_rules}"
                full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {prompt}"
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container:
                        st.markdown(response.text)
                
    with col3:
        if st.button("Phương pháp NC"):
            with st.spinner("AI đang thiết kế Chương 2..."):
                query = "Đối tượng nghiên cứu, tiêu chuẩn nhận loại trừ, thiết kế nghiên cứu, cỡ mẫu, phương pháp thu thập số liệu"
                context = build_context(query, k=4)
                prompt = f"""
                Viết "Chương 2. ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU". Không dùng Heading 1 (#).
                Mục 2.2.3 BẮT BUỘC kẻ Bảng Markdown 5 cột (TT | Tên chỉ tiêu | Định nghĩa | Phân loại | Kỹ thuật thu thập). 
                {citation_rules}
                """
                full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {prompt}"
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container:
                        st.markdown(response.text)
                
    with col4:
        if st.button("Viết Bàn luận toàn diện"):
            if not my_research_data:
                st.warning("Anh cần nhập số liệu của mình vào ô 'Bộ nhớ Số liệu' trước!")
            else:
                with st.spinner("AI đang viết Bàn luận phân đoạn theo tiêu đề chuẩn..."):
                    # Dùng chính số liệu của anh làm từ khóa tìm kiếm ngữ nghĩa trong Vector DB
                    context = build_context(my_research_data, k=8)
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
                       - Lồng ghép so sánh, đối chiếu trực tiếp (cao hơn, thấp hơn, tương đồng) với số liệu của các tác giả trong TÀI LIỆU Y VĂN ngay trong cùng đoạn văn đó.
                    5. Văn phong chuyên khảo y khoa hàn lâm, logic, không dùng từ ngữ cảm xúc.
                    {citation_rules}
                    """
                    full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {prompt}"
                    response = safe_generate_content(full_prop)
                    if response:
                        with ket_qua_container:
                            st.markdown(response.text) 
                        
    with col5:
        if st.button("So sánh NC liên quan"):
            if not my_research_data:
                st.warning("Anh cần nhập số liệu của mình vào ô 'Bộ nhớ Số liệu' trước!")
            else:
                with st.spinner("AI đang đối chiếu Y văn và viết phần 4.2, 4.3, 4.4..."):
                    context = build_context(my_research_data, k=8)
                    prompt = f"""
                    KẾT QUẢ NGHIÊN CỨU THỰC TẾ CỦA TÔI:
                    {my_research_data}
                    
                    YÊU CẦU ĐẦU RA BẮT BUỘC (Trình bày đúng các cấu trúc tiểu mục sau, không dùng Heading 1 hoặc 2):
                    ### 4.2.2. So sánh với các nghiên cứu và khuyến cáo
                    (Lấy số liệu của TÔI làm gốc. Trích xuất thông tin từ TÀI LIỆU Y VĂN để đối chiếu trực tiếp (cao hơn, thấp hơn, hay tương đồng). BẮT BUỘC giải thích sâu sắc nguyên nhân của sự khác biệt dựa trên: cỡ mẫu, đặc thù kỹ thuật, phương pháp, sự tuân thủ khuyến cáo).
                    
                    ### 4.3. Ý nghĩa lâm sàng và thực tiễn
                    (Rút ra bài học từ nghiên cứu này. Đề xuất các thay đổi thực tiễn để tối ưu hóa quy trình, giảm chi phí, nâng cao hiệu quả điều trị).
                    
                    ### 4.4. Hạn chế của nghiên cứu
                    (Tự đưa ra 2-3 hạn chế logic về cỡ mẫu, thời gian, phương pháp hồi cứu...).
                    
                    {citation_rules}
                    """
                    full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {prompt}"
                    response = safe_generate_content(full_prop)
                    if response:
                        with ket_qua_container:
                            st.markdown(response.text)
                
    with col6:
        if st.button("Trích dẫn TLTK"):
            with st.spinner("AI đang lập danh mục..."):
                query = "Tài liệu tham khảo, References, tên tác giả, năm xuất bản"
                context = build_context(query, k=10)
                prompt = f"Trích xuất tên các tác giả và bài báo có trong TÀI LIỆU Y VĂN, lập danh mục Tài liệu tham khảo chuẩn Vancouver. Không dùng Heading 1 (#)."
                full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {prompt}"
                response = safe_generate_content(full_prop)
                if response:
                    with ket_qua_container:
                        st.markdown(response.text)
    
    st.write("---")
    custom_prompt = st.text_area("Nhập câu lệnh khác ở đây:")
    if st.button("Chạy lệnh"):
        if custom_prompt:
            with st.spinner("AI đang xử lý..."):
                context = build_context(custom_prompt, k=6)
                anti_hallucination = "\nLƯU Ý NGHIÊM NGẶT: Không tự bịa thông tin. Chỉ dùng dữ liệu đã cung cấp, trích dẫn số liệu cụ thể."
                full_prop = f"TÀI LIỆU Y VĂN:\n{context}\n\nYêu cầu: {custom_prompt}\n{anti_hallucination}\n{citation_rules}"
                response = safe_generate_content(full_prop)
                if response:
                    st.markdown(response.text)
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
