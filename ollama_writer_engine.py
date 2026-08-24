# ollama_writer_engine.py
# ============================================================
# MODULE VIẾT LUẬN VĂN BẰNG OLLAMA (OFFLINE LOCAL AI)
# ============================================================

import streamlit as st
import requests
import io
from docx import Document

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def call_ollama(prompt: str, model: str, temperature: float = 0.3) -> str:
    """Gọi Ollama chạy cục bộ trên máy tính không cần Internet"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300) # Timeout dài vì AI cần thời gian viết
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"❌ Lỗi từ Ollama Server (Mã lỗi: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return "❌ Không thể kết nối tới Ollama. Hãy đảm bảo phần mềm Ollama trên máy tính đang chạy."
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
    """Giao diện Tab 5: Viết luận văn bằng Ollama"""
    st.subheader("🤖 5. Trợ lý Viết luận văn bằng Ollama (Offline)")
    st.info("Ollama sẽ lấy bản tóm tắt y văn ở Tab 3 để làm ngữ cảnh và tự động viết thành các đoạn văn bản học thuật hoàn chỉnh.")

    # Kiểm tra xem đã có bản tóm tắt từ Tab 3 chưa
    summary_context = st.session_state.get("cached_summary", "")
    if not summary_context:
        st.warning("⚠️ Chưa có dữ liệu tóm tắt! Anh hãy quay lại Tab 3 bấm nút 'Tổng hợp' trước để chuẩn bị ngữ cảnh cho AI.")
        return

    col1, col2 = st.columns(2)
    with col1:
        ollama_model = st.selectbox("Lựa chọn Model:", ["qwen2.5:7b", "qwen2.5:14b", "llama3:8b"], key="ollama_model_select")
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

    if st.button("🚀 Yêu cầu Ollama Viết Bài", type="primary", key="btn_run_ollama_writer"):
        with st.spinner(f"Ollama đang vắt óc viết phần '{section_choice}' (Có thể mất 1-3 phút tùy cấu hình máy)..."):
            # Lắp ráp câu lệnh Prompt chuẩn y khoa
            prompt = f"""
Bạn là một chuyên gia Dược lâm sàng đang viết luận văn Chuyên khoa cấp I.
NHIỆM VỤ: Hãy viết phần '{section_choice}' dựa trên các tài liệu tóm tắt dưới đây.
QUY TẮC: 
1. Văn phong học thuật, khách quan, rõ ràng.
2. Tuyệt đối không bịa đặt số liệu ngoài tài liệu được cung cấp.
3. {extra_prompt}

NGUỒN TÀI LIỆU TÓM TẮT ĐỂ THAM KHẢO:
{summary_context}
"""
            # Gọi AI
            result = call_ollama(prompt, model=ollama_model)

            # Hiển thị kết quả
            st.markdown("---")
            st.markdown("### 📝 Bản thảo từ Ollama:")
            st.markdown(result)
            
            # Tính năng tải file Word
            st.download_button(
                label="📥 Tải Bản thảo ra file Word (.docx)",
                data=create_word_document(f"Bản thảo - {section_choice}", result),
                file_name=f"Ban_thao_{section_choice}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_word_ollama"
            )
