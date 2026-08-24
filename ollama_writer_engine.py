# ollama_writer_engine.py
# ============================================================
# MODULE VIẾT LUẬN VĂN BẰNG OLLAMA (KẾT NỐI GOOGLE COLAB / OFFLINE)
# ============================================================

import streamlit as st
import requests
import io
from docx import Document

def call_ollama_colab(prompt: str, url: str, model: str, temperature: float = 0.3) -> str:
    """Gửi prompt đến máy chủ Ollama trên Google Colab qua Cloudflare Tunnel"""
    if not url:
        return "⚠️ Báo lỗi: Vui lòng dán đường dẫn Cloudflare (.trycloudflare.com) từ Google Colab vào thanh bên (Sidebar) bên trái trước khi bấm gọi AI!"
    
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
        # Timeout 180 giây để đảm bảo GPU T4 có đủ thời gian xử lý các đoạn văn dài
        response = requests.post(api_endpoint, json=payload, timeout=180)
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
    """Giao diện Tab: Viết luận văn bằng Ollama (Colab Backend)"""
    st.subheader("🤖 Trợ lý Viết luận văn bằng Ollama (Google Colab GPU)")
    st.info("Hệ thống sẽ lấy bản tóm tắt y văn ở Tab 4 làm ngữ cảnh kết hợp với sức mạnh GPU T4 để tự động viết các đoạn văn bản học thuật.")

    # --- THANH BÊN (SIDEBAR): Ô NHẬP LINK CLOUDFLARE ---
    st.sidebar.divider()
    st.sidebar.header("🔌 Kết nối Máy chủ AI (Colab)")
    st.sidebar.info("💡 Copy link .trycloudflare.com từ Colab dán vào đây")
    
    ollama_url = st.sidebar.text_input(
        "API Base URL:", 
        placeholder="https://...trycloudflare.com",
        key="ollama_colab_url_tab6"
    )

    # Kiểm tra xem đã có bản tóm tắt từ Tab Tóm tắt chưa
    summary_context = st.session_state.get("cached_summary", "")
    if not summary_context:
        st.warning("⚠️ Chưa có dữ liệu tóm tắt! Anh hãy quay lại Tab '4. Tóm tắt' bấm nút 'Tổng hợp' trước để chuẩn bị ngữ cảnh cho AI.")
        return

    col1, col2 = st.columns(2)
    with col1:
        # Ưu tiên đặt qwen2.5:14b lên đầu vì chúng ta chạy trên GPU 16GB của Colab
        ollama_model = st.selectbox(
            "Lựa chọn Model trên Colab:", 
            ["qwen2.5:14b", "qwen2.5:7b", "llama3:8b", "qwen2.5:3b"], 
            key="ollama_model_select"
        )
    with col2:
        section_choice = st.selectbox(
            "Chọn phần cần viết:",
            ["Đặt vấn đề", "Tổng quan tài liệu", "Bàn luận chuyên sâu", "Kết luận"],
            key="ollama_section_select"
        )

    extra_prompt = st.text_area(
        "Ghi chú thêm cho AI (Tùy chọn):", 
        placeholder="VD: Hãy viết khoảng 500 từ, nhấn mạnh vào cơ chế tác dụng của kháng sinh...", 
        key="ollama_extra_prompt"
    )

    if st.button("🚀 Yêu cầu Ollama (Colab) Viết Bài", type="primary", key="btn_run_ollama_writer"):
        with st.spinner(f"GPU T4 trên Colab đang xử lý phần '{section_choice}'..."):
            
            # Xây dựng prompt động theo lựa chọn của anh
            prompt = f"""Hãy đóng vai một nhà nghiên cứu Dược lâm sàng chuyên nghiệp viết phần '{section_choice}' cho luận văn chuyên khoa.
Dựa vào các thông tin tóm tắt y văn bên dưới, hãy viết một đoạn văn bản học thuật hoàn chỉnh, chuẩn mực y khoa.

THÔNG TIN NỀN TẢNG THAM KHẢO:
{summary_context}

GHI CHÚ THÊM TỪ NGƯỜI DÙNG:
{extra_prompt if extra_prompt else "Không có"}

YÊU CẦU: Viết văn xuôi mạch lạc, học thuật, khách quan, không chào hỏi, không gạch đầu dòng rườm rà.
"""
            # Gọi hàm qua Colab
            result = call_ollama_colab(prompt=prompt, url=ollama_url, model=ollama_model)

            # Hiển thị kết quả
            st.markdown("---")
            st.markdown("### 📝 Bản thảo từ Ollama (Colab GPU):")
            st.markdown(
                f"<div style='background-color: white; padding: 20px; border-radius: 10px; color: black; border: 1px solid #ddd; font-size: 16px;'>"
                f"{result}</div>", 
                unsafe_allow_html=True
            )
            
            # Tính năng tải file Word
            st.download_button(
                label="📥 Tải Bản thảo ra file Word (.docx)",
                data=create_word_document(f"Bản thảo - {section_choice}", result),
                file_name=f"Ban_thao_{section_choice}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_word_ollama"
            )
