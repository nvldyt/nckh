# ollama_writer_engine.py
# ============================================================
# MODULE VIẾT LUẬN VĂN BẰNG OLLAMA (RAG TRỰC TIẾP TỪ KHO TÀI LIỆU GỐC)
# ============================================================

import streamlit as st
import requests
import io
import re
from docx import Document

try:
    from retrieval_engine import retrieve_evidence
except ImportError:
    retrieve_evidence = None

def call_ollama_colab(prompt: str, url: str, model: str, temperature: float = 0.3) -> str:
    """Gửi prompt đến máy chủ Ollama trên Google Colab qua Cầu nối FastAPI"""
    if not url:
        return "⚠️ Báo lỗi: Vui lòng dán đường dẫn Cloudflare (.trycloudflare.com) từ Google Colab vào thanh bên (Sidebar) bên trái trước khi bấm gọi AI!"
    
    # Tự động làm sạch URL (phòng hờ dính Markdown hoặc khoảng trắng)
    url = url.strip()
    md_match = re.search(r'\((https?://[^\s)]+)\)', url)
    if md_match:
        url = md_match.group(1)
    else:
        url = url.strip("[]'\"<>")

    api_endpoint = f"{url.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        # Timeout 300 giây để mô hình 14B có đủ thời gian suy luận văn bản dài
        response = requests.post(api_endpoint, json=payload, timeout=300)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"❌ Lỗi từ máy chủ Colab (Mã lỗi: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return "❌ Mất kết nối với Colab. Hãy kiểm tra lại đường link Cloudflare hoặc xem Colab đã bị tắt chưa."
    except Exception as exc:
        return f"❌ Lỗi kết nối: {exc}"

def create_word_document(title: str, body: str) -> bytes:
    """Tạo file Word (.docx) từ văn bản kết quả"""
    doc = Document()
    doc.add_heading(title, level=0)
    for line in body.split("\n"):
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith(("- ", "* ")):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        else:
            if line.strip(): 
                doc.add_paragraph(line.strip())
    
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def render_ollama_writer_tab():
    """Giao diện Tab 6: Viết luận văn bằng Ollama tích hợp RAG trực tiếp"""
    st.subheader("🤖 Trợ lý Viết luận văn bằng Ollama (RAG Trực tiếp từ Kho tài liệu)")
    st.info("💡 Hệ thống sẽ tự động quét kho tài liệu gốc (PDF/PubMed), truy xuất các đoạn bằng chứng xác thực nhất và chuyển cho Ollama (Colab GPU) viết bài.")

    # --- THANH BÊN (SIDEBAR): Ô NHẬP LINK CLOUDFLARE ---
    st.sidebar.divider()
    st.sidebar.header("🔌 Kết nối Máy chủ AI (Colab)")
    st.sidebar.info("💡 Copy link .trycloudflare.com từ Colab dán vào đây")
    
    ollama_url = st.sidebar.text_input(
        "API Base URL:", 
        placeholder="https://...trycloudflare.com",
        key="ollama_colab_url_tab6"
    )

    # Kiểm tra xem Evidence Database đã có tài liệu chưa
    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.warning("⚠️ Evidence Database đang trống! Hãy nạp tài liệu PDF (Tab 1) hoặc bài báo PubMed (Tab 2) trước để Ollama có dữ liệu viết bài.")
        return

    col1, col2 = st.columns(2)
    with col1:
        ollama_model = st.selectbox(
            "Lựa chọn Model trên Colab:", 
            ["qwen2.5:14b", "qwen2.5:7b", "llama3:8b", "qwen2.5:3b"], 
            key="ollama_model_select"
        )
    with col2:
        top_k = st.slider("Số đoạn bằng chứng truy xuất:", 3, 15, 8, key="ollama_top_k")

    user_query = st.text_area(
        "Nhập yêu cầu viết văn hoặc vấn đề cần phân tích học thuật:", 
        placeholder="VD: Viết phần Đặt vấn đề và tính cấp thiết của việc theo dõi nồng độ Vancomycin trên bệnh nhân suy thận...", 
        height=120,
        key="ollama_user_query"
    )

    if st.button("🚀 Truy xuất bằng chứng & Yêu cầu Ollama Viết Bài", type="primary", key="btn_run_ollama_writer"):
        if not user_query.strip():
            st.warning("⚠️ Vui lòng nhập nội dung yêu cầu trước khi gửi.")
        else:
            with st.spinner("Đang truy xuất bằng chứng từ kho tài liệu và gọi GPU Colab xử lý..."):
                
                # 1. Truy xuất bằng chứng trực tiếp từ kho tài liệu gốc
                evidence = []
                if retrieve_evidence:
                    try:
                        evidence = retrieve_evidence(
                            query=user_query,
                            chunks=chunks,
                            matrix=st.session_state.get("embeddings"),
                            bm25=st.session_state.get("bm25"),
                            top_k=top_k
                        )
                    except Exception as e:
                        st.error(f"❌ Lỗi truy xuất bằng chứng: {e}")
                
                if not evidence:
                    st.warning("⚠️ Không tìm thấy đoạn bằng chứng phù hợp trong kho tài liệu.")
                    return

                # 2. Xây dựng ngữ cảnh từ các đoạn bằng chứng (có kèm mã nguồn SRC-...)
                evidence_blocks = []
                for ev in evidence:
                    src_id = ev.get("source_id", "UNKNOWN")
                    chunk_id = ev.get("chunk_id", "CHUCK")
                    text = ev.get("text", "")
                    evidence_blocks.append(f"[{src_id} - {chunk_id}]: {text}")
                
                evidence_text = "\n\n".join(evidence_blocks)

                # 3. Lắp ráp Prompt chuẩn y khoa nâng cao
                prompt = f"""Bạn là một chuyên gia nghiên cứu và giảng viên Dược lâm sàng hàng đầu. Nhiệm vụ của bạn là viết một phần nội dung luận văn chuyên khoa cấp I dựa trên các bằng chứng khoa học thực tế được cung cấp dưới đây.

YÊU CẦU / CHỦ ĐỀ CỦA NGƯỜI DÙNG:
{user_query}

DANH MỤC BẰNG CHỨNG KHOA HỌC THỰC TẾ (BẠN PHẢI DỰA VÀO ĐÂY ĐỂ VIẾT):
{evidence_text}

QUY TẮC VIẾT VĂN BẮC BUỘC (TUÂN THỦ 100%):
1. Viết hoàn toàn bằng VĂN XUÔI LIỀN MẠCH, chia thành các đoạn văn học thuật rõ ràng (mỗi đoạn 5-7 câu). 
2. TUYỆT ĐỐI KHÔNG dùng cấu trúc kiểu tóm tắt bài báo (không dùng các nhãn cứng nhắc như 'Mục tiêu', 'Phương pháp', 'Kết quả', 'Kết luận').
3. Phải đính kèm mã nguồn dạng [SRC-XXXX] ngay sau các nhận định, dữ liệu lấy từ bằng chứng.
4. Phân tích sâu sắc về mặt dược động học, dược lực học (PK/PD) hoặc ý nghĩa lâm sàng, văn phong trang trọng, chuẩn mực của luận văn y khoa.
"""

                # 4. Gọi Ollama qua Colab
                result = call_ollama_colab(prompt=prompt, url=ollama_url, model=ollama_model)

                # 5. Hiển thị kết quả
                st.markdown("---")
                st.markdown("### 📝 Bản thảo từ Ollama (RAG Trực tiếp):")
                st.markdown(
                    f"<div style='background-color: white; padding: 20px; border-radius: 10px; color: black; border: 1px solid #ddd; font-size: 16px;'>"
                    f"{result}</div>", 
                    unsafe_allow_html=True
                )
                
                # 6. Hiển thị dấu vết bằng chứng (Evidence Trace) để đối chiếu
                with st.expander("🔎 Xem dấu vết bằng chứng (Evidence Trace) đã sử dụng"):
                    for ev in evidence:
                        meta = st.session_state.get("documents", {}).get(ev.get("source_id"), {})
                        st.markdown(f"- **Tài liệu:** `{meta.get('file_name', 'N/A')}` | **Độ khớp:** `{ev.get('score', 0):.4f}` | **Mã đoạn:** `{ev.get('chunk_id', '')}`")
                        st.info(f"_{ev.get('text', '')}_")

                # 7. Nút tải file Word
                st.download_button(
                    label="📥 Tải Bản thảo ra file Word (.docx)",
                    data=create_word_document("Bản thảo nghiên cứu khoa học", result),
                    file_name="Ban_thao_RAG_Ollama.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_word_ollama_rag"
                )
