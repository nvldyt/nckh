import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import time
from google.api_core.exceptions import ResourceExhausted

# THƯ VIỆN CHO TAB TRA CỨU ĐA NGUỒN (PubMed + Tạp chí Y học VN)
import requests
import xml.etree.ElementTree as ET
import io
from docx import Document

# Danh sách domain tạp chí Y học Việt Nam - ANH TỰ CHỈNH LẠI CHO ĐÚNG NẾU CẦN
VN_JOURNAL_DOMAINS = [
    "tapchiyhocvietnam.vn",
    "vjol.info",
    "tapchinghiencuuyhoc.vn",
    "jmp.huemed-univ.edu.vn",
]

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

    /* ===== NỘI DUNG BÀI VIẾT DO AI TẠO - CHỮ NHỎ, CANH ĐỀU 2 LỀ, DỄ ĐỌC ===== */
    .stMarkdown p, .stMarkdown li {
        font-size: 0.92rem !important;
        line-height: 1.75 !important;
        text-align: justify !important;
        text-justify: inter-word;
    }
    .stMarkdown table td, .stMarkdown table th {
        font-size: 0.85rem !important;
    }
    /* Giữ tiêu đề (h1,h2,h3) không bị justify/canh trái đều, chỉ áp cho đoạn văn */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        text-align: left !important;
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
    
    # MODEL CHÍNH: dùng cho các tác vụ cần suy luận sâu, độ chính xác cao
    # (viết Đặt vấn đề, Tổng quan, Bàn luận, Phương pháp NC, trích dẫn TLTK...)
    model = genai.GenerativeModel("gemini-3.7-flash", system_instruction=system_prompt)
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    
    # MODEL NHẸ - TỐC ĐỘ CAO: chỉ dùng để tóm tắt/rút trích PDF (tác vụ đơn giản, khối lượng lớn)
    # Nhanh hơn, rẻ hơn, và thường có hạn mức miễn phí (quota) cao hơn model chính
    model_extract = genai.GenerativeModel("gemini-3.5-flash-lite", system_instruction=system_prompt)
    generation_config_extract = genai.types.GenerationConfig(temperature=0.1)
    
    # HÀM BỌC AN TOÀN: Tự động né lỗi quá tải API
    def safe_generate_content(prompt, config=generation_config, max_retries=10, use_model=None):
        target_model = use_model if use_model is not None else model
        for attempt in range(max_retries):
            try:
                return target_model.generate_content(prompt, generation_config=config)
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
# CÁC HÀM DÙNG CHO TAB TRA CỨU ĐA NGUỒN
# (PubMed quốc tế + Tạp chí Y học Việt Nam)
# ==========================================
def translate_and_optimize_query(vietnamese_query: str) -> str:
    prompt = (
        f"Chuyển đổi từ khóa tiếng Việt sau thành chuỗi từ khóa y khoa (MeSH terms) "
        f"bằng tiếng Anh tối ưu nhất để tìm trên PubMed.\n"
        f"Từ gốc: {vietnamese_query}\n"
        f"Chỉ trả về chuỗi tiếng Anh, không giải thích."
    )
    response = safe_generate_content(prompt)
    if response and hasattr(response, 'text'):
        return response.text.strip().replace('"', '')
    return vietnamese_query


def fetch_pubmed_details(id_list: list):
    if not id_list:
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
    res = requests.get(url, params=params)
    articles = []
    if res.status_code == 200:
        root = ET.fromstring(res.content)
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            title_node = article.find(".//ArticleTitle")
            pmid = pmid_node.text if pmid_node is not None else "Unknown"
            title = title_node.text if title_node is not None else "Unknown Title"

            abstracts = article.findall(".//AbstractText")
            abs_text = " ".join([e.text for e in abstracts if e.text])

            author_node = article.find(".//Author/LastName")
            author = author_node.text if author_node is not None else "Unknown"
            year_node = article.find(".//PubDate/Year")
            year = year_node.text if year_node is not None else "Unknown"

            articles.append({
                "id": pmid,
                "title": title,
                "abstract": abs_text if abs_text else "Không có bản tóm tắt (No abstract).",
                "citation": f"{author} et al., {year}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
    return articles


def search_vn_medical_journals(vietnamese_query: str, max_res: int = 5):
    """
    Tìm bài báo tiếng Việt trên các tạp chí Y học VN thông qua Google Scholar
    (qua SerpAPI). Nếu VN_JOURNAL_DOMAINS rỗng, mở rộng tìm kiếm với site:.vn.
    """
    serpapi_key = st.secrets.get("SERPAPI_KEY", "")
    if not serpapi_key:
        return [], "Chưa cấu hình SERPAPI_KEY trong Streamlit Secrets."

    if VN_JOURNAL_DOMAINS:
        domain_filter = " OR ".join([f"site:{d}" for d in VN_JOURNAL_DOMAINS])
        full_query = f'{vietnamese_query} tạp chí y học ({domain_filter})'
    else:
        full_query = f'{vietnamese_query} tạp chí y học site:.vn'

    params = {
        "engine": "google_scholar",
        "q": full_query,
        "hl": "vi",
        "num": max_res,
        "api_key": serpapi_key,
    }

    try:
        res = requests.get("https://serpapi.com/search", params=params, timeout=20)
        data = res.json()
    except Exception as e:
        return [], f"Lỗi kết nối SerpAPI: {e}"

    results = []
    for item in data.get("organic_results", [])[:max_res]:
        results.append({
            "title": item.get("title", "Không có tiêu đề"),
            "link": item.get("link", "#"),
            "snippet": item.get("snippet", "Không có đoạn trích."),
            "source": item.get("publication_info", {}).get("summary", "Không rõ nguồn"),
        })
    return results, None


def generate_word_document(content_text, heading="Tổng hợp Tài liệu Y văn"):
    doc = Document()
    doc.add_heading(heading, 0)
    doc.add_paragraph(content_text)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# ==========================================
# CÁC TAB CHỨC NĂNG
# ==========================================
tab3, tab1, tab2, tab4 = st.tabs([
    "🔬 Tra cứu Đa nguồn (PubMed + Tạp chí VN)",
    "📄 Đọc Tài liệu & Viết Luận văn (RAG)",
    "📊 Phân tích Số liệu Bệnh án (Excel)",
    "🔍 Kiểm tra & Audit bài viết",
])

# ----------------------------------------------------
# TAB 1: RAG VECTOR DATABASE
# ----------------------------------------------------
with tab1:
    st.header("Phân tích tài liệu và Viết bài")
    
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "ngan_hang_y_van" not in st.session_state:
        st.session_state["ngan_hang_y_van"] = ""
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "last_success" not in st.session_state:
        st.session_state["last_success"] = None
        
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
                if not uploaded_files:
                    st.session_state["last_error"] = "Vui lòng tải lên ít nhất 1 file PDF."
                    st.session_state["last_success"] = None
                else:
                    progress_bar = st.progress(0, text="Chuẩn bị xử lý...")
                    status_text = st.empty()
                    total_files = len(uploaded_files)
                    ket_qua_gop = ""
                    loi_gop = []

                    for idx, uploaded_file in enumerate(uploaded_files, start=1):
                        progress_bar.progress(
                            idx / total_files,
                            text=f"Đang xử lý file {idx}/{total_files}: {uploaded_file.name}"
                        )

                        # Đọc riêng từng file PDF
                        file_text = ""
                        try:
                            reader = PdfReader(uploaded_file)
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    file_text += page_text + "\n"
                        except Exception as e:
                            loi_gop.append(f"{uploaded_file.name}: lỗi đọc file ({e})")
                            continue

                        if not file_text.strip():
                            loi_gop.append(f"{uploaded_file.name}: không đọc được chữ (có thể là file scan/ảnh)")
                            continue

                        text_to_send = file_text[:60000]

                        extract_prompt = f"""
                        Đây là nội dung trích từ file "{uploaded_file.name}".
                        Hãy đọc và TÓM TẮT CÔ ĐẶC lại những thông tin sau:
                        1. Tên tác giả, năm nghiên cứu, tên bài báo.
                        2. Mục tiêu nghiên cứu và đối tượng nghiên cứu.
                        3. Các kết quả, số liệu quan trọng nhất (tỷ lệ %, p-value, OR, RR...).
                        4. Kết luận chính của tác giả.
                        Tuyệt đối không bịa số liệu. Trình bày dưới dạng gạch đầu dòng ngắn gọn.
                        """
                        full_prop = f"Tài liệu gốc:\n{text_to_send}\n\nYêu cầu: {extract_prompt}"

                        # Dùng model NHẸ (Flash-Lite) cho tác vụ tóm tắt - nhanh hơn, ít bị quá tải hơn
                        response = safe_generate_content(
                            full_prop,
                            config=generation_config_extract,
                            max_retries=3,
                            use_model=model_extract
                        )

                        if response and response.text:
                            ket_qua_gop += f"\n\n--- Nguồn: {uploaded_file.name} ---\n{response.text}"
                        else:
                            loi_gop.append(f"{uploaded_file.name}: không nhận được kết quả từ AI sau nhiều lần thử")

                        status_text.empty()
                        # Nghỉ ngắn giữa các file - Flash-Lite nhanh & nhẹ nên không cần nghỉ lâu như model chính
                        time.sleep(3)

                    progress_bar.empty()

                    if ket_qua_gop:
                        st.session_state["ngan_hang_y_van"] += ket_qua_gop
                        st.session_state["last_success"] = f"✅ Đã rút trích xong {total_files - len(loi_gop)}/{total_files} file thành công (dùng model Flash-Lite tốc độ cao)."
                    else:
                        st.session_state["last_success"] = None

                    if loi_gop:
                        st.session_state["last_error"] = "⚠️ Một số file gặp lỗi:\n" + "\n".join(loi_gop)
                    else:
                        st.session_state["last_error"] = None

                st.rerun()

        # Hiển thị lỗi/thành công đã lưu lại XUYÊN SUỐT qua rerun (không bị mất)
        if st.session_state.get("last_error"):
            st.error(st.session_state["last_error"])
        if st.session_state.get("last_success"):
            st.success(st.session_state["last_success"])
        
        # Ô hiển thị + cho phép chỉnh sửa Ngân hàng y văn đã rút trích
        st.text_area(
            "📚 Ngân hàng y văn đã rút trích (có thể sửa tay trước khi dùng):",
            height=200,
            key="ngan_hang_y_van"
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
    def retrieve_context(query, k=5):
        if st.session_state.get("vector_store") is not None:
            docs = st.session_state["vector_store"].similarity_search(query, k=k)
            return "\n\n".join([f"--- Đoạn trích ---:\n{d.page_content}" for d in docs])
        return "Không có dữ liệu y văn trong Vector Database."

    def build_context(query, k=6):
        vector_context = retrieve_context(query, k=k)
        ngan_hang = st.session_state.get("ngan_hang_y_van", "").strip()
        
        parts = []
        if vector_context and "Không có dữ liệu" not in vector_context:
            parts.append(f"[TRÍCH ĐOẠN GỐC TỪ VECTOR DATABASE - PDF]:\n{vector_context}")
        if ngan_hang:
            parts.append(f"[TÓM TẮT TỪ NGÂN HÀNG Y VĂN]:\n{ngan_hang}")
        
        if not parts:
            return "Không có dữ liệu y văn nào được cung cấp. Hãy trả lời: 'Tài liệu không đề cập'."
        return "\n\n".join(parts)
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
                progress_bar = st.progress(0, text="Chuẩn bị...")
                total = len(vars_desc)
                for i, var in enumerate(vars_desc, start=1):
                    progress_bar.progress(i / total, text=f"Đang xử lý biến {i}/{total}: {var}")
                    
                    freq_table = df[var].value_counts().to_string()
                    total_n = len(df[var].dropna())
                    
                    prompt = f"""
                    Dữ liệu đếm thực tế của biến '{var}': {freq_table} (Tổng: {total_n}).
                    Yêu cầu: 1. Vẽ bảng SPSS (Phân loại, n, %). 2. Viết nhận xét y khoa chuyên sâu, khô khan.
                    """
                    response = safe_generate_content(
                        prompt,
                        config=generation_config_extract,
                        max_retries=3,
                        use_model=model_extract
                    )
                    
                    if response:
                        st.subheader(f"► Phân tích biến: {var}")
                        st.markdown(response.text)
                        st.write("---")
                    else:
                        st.error(f"⚠️ Không lấy được kết quả cho biến '{var}' (quá tải API sau 3 lần thử).")
                
                progress_bar.empty()
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
                bien_hop_le = [v for v in indep_cols if v != target_col]
                progress_bar = st.progress(0, text="Chuẩn bị...")
                total = len(bien_hop_le)
                
                for i, var in enumerate(bien_hop_le, start=1):
                    progress_bar.progress(i / total, text=f"Đang xử lý bảng chéo {i}/{total}: {var} × {target_col}")
                    
                    crosstab_df = pd.crosstab(df[var], df[target_col])
                    
                    prompt = f"""
                    Bảng Crosstabs thực tế giữa '{var}' và '{target_col}':\n{crosstab_df.to_string()}\n
                    Yêu cầu: 1. Trình bày bảng khoa học (% hàng/cột). 2. Nhận xét chuyên sâu.
                    """
                    response = safe_generate_content(
                        prompt,
                        config=generation_config_extract,
                        max_retries=3,
                        use_model=model_extract
                    )
                    
                    if response:
                        st.subheader(f"► Mối liên quan giữa {var} và {target_col}")
                        st.markdown(response.text)
                        st.write("---")
                    else:
                        st.error(f"⚠️ Không lấy được kết quả cho '{var} × {target_col}' (quá tải API sau 3 lần thử).")
                
                progress_bar.empty()
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

# ----------------------------------------------------
# TAB 3: TRA CỨU ĐA NGUỒN (PubMed + Tạp chí Y học Việt Nam)
#        + TỰ ĐỘNG TỔNG HỢP TÓM TẮT
# ----------------------------------------------------
with tab3:
    st.header("🔬 Tra cứu Đa nguồn: PubMed (Quốc tế) + Tạp chí Y học Việt Nam")
    st.info(
        "💡 Chỉ cần nhập **tên đề tài bằng tiếng Việt**. Hệ thống sẽ tự động dịch "
        "sang từ khoá MeSH để tìm trên PubMed, đồng thời tìm các bài báo liên quan "
        "trên tạp chí Y học Việt Nam, rồi tổng hợp thành một bản tóm tắt duy nhất."
    )

    # --- Khởi tạo session state riêng cho Tab 3 ---
    for key, default in [
        ("t3_pm_data", []), ("t3_vn_data", []), ("t3_en_keyword", ""),
        ("t3_summary", ""), ("t3_query", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    if "saved_reviews" not in st.session_state:
        st.session_state["saved_reviews"] = []

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        t3_query = st.text_input(
            "Nhập tên đề tài nghiên cứu (Tiếng Việt):",
            placeholder="VD: Hiệu quả kiểm soát đường huyết bằng metformin ở bệnh nhân đái tháo đường type 2",
            key="t3_query_input",
        )
    with col_btn:
        max_res = st.number_input("Số bài/nguồn", min_value=2, max_value=10, value=5, key="t3_max_res")

    btn_search = st.button("🚀 Tra cứu song song 2 nguồn", type="primary", key="t3_btn_search")

    if btn_search:
        if not t3_query:
            st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
        else:
            st.session_state["t3_query"] = t3_query

            with st.spinner("🧠 AI đang dịch & chuẩn hoá từ khoá sang MeSH (tiếng Anh)..."):
                en_query = translate_and_optimize_query(t3_query)
                st.session_state["t3_en_keyword"] = en_query

            with st.spinner("🌍 Đang tìm & tải Abstract từ PubMed..."):
                search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                search_params = {"db": "pubmed", "term": en_query, "retmode": "json", "retmax": max_res}
                try:
                    id_list = requests.get(search_url, params=search_params).json() \
                        .get("esearchresult", {}).get("idlist", [])
                    st.session_state["t3_pm_data"] = fetch_pubmed_details(id_list)
                except Exception as e:
                    st.session_state["t3_pm_data"] = []
                    st.error(f"Lỗi tìm PubMed: {e}")

            with st.spinner("🇻🇳 Đang tìm bài báo trên tạp chí Y học Việt Nam..."):
                vn_results, vn_err = search_vn_medical_journals(t3_query, max_res)
                st.session_state["t3_vn_data"] = vn_results
                if vn_err:
                    st.error(vn_err)

            # Reset tóm tắt cũ khi tìm kiếm mới
            st.session_state["t3_summary"] = ""

    # --- Hiển thị kết quả 2 cột ---
    if st.session_state["t3_pm_data"] or st.session_state["t3_vn_data"]:
        st.write("---")
        col_vn, col_pm = st.columns(2)

        with col_vn:
            st.markdown("### 🇻🇳 Tạp chí Y học Việt Nam")
            if not st.session_state["t3_vn_data"]:
                st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
            else:
                for art in st.session_state["t3_vn_data"]:
                    st.markdown(f"**[{art['title']}]({art['link']})**")
                    st.caption(art["source"])
                    st.write(art["snippet"])
                    st.divider()

        with col_pm:
            st.markdown("### 🌍 PubMed (Quốc tế)")
            if st.session_state["t3_en_keyword"]:
                st.success(f"🔑 Từ khoá MeSH: **{st.session_state['t3_en_keyword']}**")
            if not st.session_state["t3_pm_data"]:
                st.info("Chưa có dữ liệu / không tìm thấy kết quả phù hợp.")
            else:
                for art in st.session_state["t3_pm_data"]:
                    st.markdown(f"**[{art['title']}]({art['url']})**")
                    st.caption(f"✍️ {art['citation']}")
                    with st.expander("Xem tóm tắt (Abstract)"):
                        st.write(art["abstract"])
                    st.divider()

        # --- Nút tổng hợp tóm tắt bằng Gemini ---
        st.write("---")
        if st.button("✍️ Tổng hợp & Tóm tắt toàn bộ (PubMed + VN)", type="primary", key="t3_btn_summarize"):
            if not st.session_state["t3_pm_data"] and not st.session_state["t3_vn_data"]:
                st.warning("Không có tài liệu nào để tổng hợp. Vui lòng tra cứu trước.")
            else:
                with st.spinner("AI đang đọc và tổng hợp nội dung chính..."):
                    context_parts = []
                    for idx, art in enumerate(st.session_state["t3_pm_data"]):
                        context_parts.append(
                            f"[PubMed {idx + 1}]\nTiêu đề: {art['title']}\n"
                            f"Tóm tắt: {art['abstract']}\nTrích dẫn: {art['citation']}"
                        )
                    for idx, art in enumerate(st.session_state["t3_vn_data"]):
                        context_parts.append(
                            f"[VN {idx + 1}]\nTiêu đề: {art['title']}\n"
                            f"Đoạn trích: {art['snippet']}\nNguồn: {art['source']}"
                        )

                    context_text = "\n\n".join(context_parts)
                    prompt = (
                        "Bạn là chuyên gia viết 'Tổng quan tài liệu' cho luận văn y khoa "
                        "(Dược sĩ Chuyên khoa Cấp I).\n"
                        f"Dựa trên các tài liệu sau (gồm cả PubMed quốc tế và tạp chí Việt Nam):\n"
                        f"{context_text}\n\n"
                        "Yêu cầu:\n"
                        "1. Tổng hợp thành một bài viết tiếng Việt mạch lạc, logic, nêu bật các "
                        "điểm chính, số liệu, và điểm tương đồng/khác biệt giữa nghiên cứu trong "
                        "và ngoài nước.\n"
                        "2. Bắt buộc chèn trích dẫn dạng [PubMed X] hoặc [VN X] ngay sau mỗi "
                        "thông tin lấy từ tài liệu tương ứng.\n"
                        "3. Văn phong hàn lâm, khô khan, trực diện, không suy diễn ngoài dữ liệu "
                        "đã cho.\n"
                        "4. Nếu đoạn trích tiếng Việt quá ngắn để kết luận chắc chắn, hãy nêu rõ "
                        "đây là thông tin sơ bộ cần kiểm tra lại bản gốc."
                    )

                    gemini_res = safe_generate_content(prompt)
                    if gemini_res and hasattr(gemini_res, 'text'):
                        st.session_state["t3_summary"] = gemini_res.text
                        st.session_state["saved_reviews"].append({
                            "query": st.session_state["t3_query"],
                            "content": gemini_res.text,
                        })
                        st.success("✅ Đã tổng hợp xong!")

    # --- Hiển thị kết quả tổng hợp + tải Word ---
    if st.session_state["t3_summary"]:
        st.write("---")
        st.subheader("📄 Bản tổng hợp")
        with st.container(border=True):
            st.markdown(st.session_state["t3_summary"])

        word_file = generate_word_document(
            st.session_state["t3_summary"],
            heading=f"Tổng hợp Tài liệu: {st.session_state['t3_query']}",
        )
        st.download_button(
            label="📥 Tải xuống file Word (.docx)",
            data=word_file,
            file_name="Tong_Hop_Da_Nguon.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="t3_download_word",
        )

# ----------------------------------------------------
# TAB 4: KIỂM TRA & AUDIT BÀI VIẾT
# ----------------------------------------------------
with tab4:
    st.header("🔍 Kiểm tra & Audit bài viết")
    st.info("💡 Tab này dùng để AI đóng vai một 'Phản biện khó tính' kiểm tra lại nội dung bạn đã viết.")
    
    text_to_check = st.text_area("Dán đoạn văn bản cần kiểm tra (Luận văn/Bàn luận):", height=300, key="t4_text_to_check")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("✅ Kiểm tra Lỗi chính tả & Y khoa", key="t4_btn_check_spelling"):
            if not text_to_check.strip():
                st.warning("Vui lòng dán đoạn văn bản cần kiểm tra trước!")
            else:
                with st.spinner("Đang rà soát thuật ngữ..."):
                    prompt = f"""
                    Bạn là biên tập viên y khoa khó tính. Hãy kiểm tra đoạn văn sau:
                    1. Phát hiện lỗi chính tả, lỗi dùng từ chuyên ngành.
                    2. Kiểm tra tính nhất quán của các trích dẫn số [x].
                    3. Đề xuất cách diễn đạt hàn lâm hơn cho các câu bị lủng củng.
                    Đoạn văn: {text_to_check}
                    """
                    response = safe_generate_content(prompt)
                    if response: st.markdown(response.text)
    with col_b:
        if st.button("⚖️ Kiểm tra Logic & Đạo văn (Audit)", key="t4_btn_check_logic"):
            if not text_to_check.strip():
                st.warning("Vui lòng dán đoạn văn bản cần kiểm tra trước!")
            else:
                with st.spinner("Đang phân tích cấu trúc..."):
                    prompt = f"""
                    Hãy đóng vai người phản biện luận văn. Kiểm tra đoạn văn sau:
                    1. Độ logic: Các lập luận có bị vòng vo hay mâu thuẫn không?
                    2. Độ tin cậy: Có câu nào nghe như 'bịa đặt' hoặc thiếu căn cứ không?
                    3. Tính đạo văn tiềm ẩn: Đoạn văn này có bị lặp cấu trúc câu quá nhiều hay nghe giống văn phong 'AI tạo sinh' (robot) không?
                    Đoạn văn: {text_to_check}
                    """
                    response = safe_generate_content(prompt)
                    if response: st.markdown(response.text)
    with col_c:
        if st.button("🎓 Văn phong học thuật (Re-write)", key="t4_btn_rewrite"):
            if not text_to_check.strip():
                st.warning("Vui lòng dán đoạn văn bản cần kiểm tra trước!")
            else:
                with st.spinner("Đang nâng cấp văn phong..."):
                    prompt = f"""
                    Hãy viết lại đoạn văn sau theo văn phong của một bài báo khoa học y khoa (Academic Medical Journal):
                    - Khô khan, trực diện, chính xác.
                    - Loại bỏ các tính từ chỉ cảm xúc hoặc từ ngữ hoa mỹ.
                    - Câu văn ngắn gọn, logic.
                    - Đảm bảo giữ nguyên các số liệu (nếu có).
                    Đoạn văn: {text_to_check}
                    """
                    response = safe_generate_content(prompt)
                    if response: st.markdown(response.text)
