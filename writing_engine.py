import os
import time
import gc
import io
import requests
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional

import key_manager

# ============================================================
# CẤU HÌNH MODEL GROQ HIỆN HÀNH (cập nhật tháng 8/2026)
#
# ĐÃ BỊ KHAI TỬ (không dùng được nữa):
#   ❌ llama3-70b-8192          (khai tử 5/2025)
#   ❌ llama3-8b-8192           (khai tử 5/2025)
#   ❌ llama-3.3-70b-versatile  (khai tử 6/2026, hết hẳn 8/2026)
#   ❌ llama-3.1-8b-instant     (khai tử 6/2026, hết hẳn 8/2026)
#
# PRODUCTION MODELS HIỆN TẠI của Groq (tháng 8/2026):
#   ✅ openai/gpt-oss-120b  — ~500 t/s, context 131K — task nặng (viết luận)
#   ✅ openai/gpt-oss-20b   — ~1000 t/s, context 131K — task nhanh/nhẹ
# ============================================================
DEFAULT_MODEL = "openai/gpt-oss-120b"
MODEL_LITE    = "openai/gpt-oss-20b"
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"

BASE_SYSTEM_RULES = """
Bạn là trợ lý nghiên cứu khoa học, hỗ trợ viết luận văn Chuyên khoa cấp I ngành Dược lâm sàng.
NGUYÊN TẮC BẮT BUỘC:
1. Tài liệu được cung cấp là nguồn bằng chứng ưu tiên duy nhất.
2. Không tự tạo số liệu, p-value, OR, RR, HR, CI95%, tỷ lệ %, liều dùng hoặc cỡ mẫu nếu không có trong bằng chứng.
3. Không tự tạo tên tác giả, năm, tên bài báo, DOI, PMID.
4. Nếu context không đủ bằng chứng, phải diễn đạt tự nhiên (VD: "Tuy nhiên, y văn hiện tại chưa đề cập...").
5. Mọi khẳng định dựa trên tài liệu phải chèn MÃ ĐỊNH DANH của tài liệu đó ngay sau câu. Ví dụ: "Tỷ lệ này là 12% [REF-001]."
6. TUYỆT ĐỐI KHÔNG tự tạo [1], [2], [3] và không dùng citation dạng tác giả-năm.
7. KHÔNG sử dụng các nhãn phân chia máy móc như "Dữ kiện (FACT):", "Diễn giải (INTERPRETATION):".
8. Dùng chính xác thuật ngữ chuyên ngành Dược lâm sàng, văn phong y khoa liền mạch.
"""

# ============================================================
# 1. HÀM GỌI AI THÔNG QUA GROQ API
# ============================================================

def call_gemini(
    prompt: str,
    model: str = None,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> Optional[str]:
    """Gọi Groq API. Tên hàm giữ nguyên `call_gemini` để không phá vỡ
    các module khác đang import hàm này."""
    model_name = model if model else DEFAULT_MODEL

    for attempt in range(max_retries):
        try:
            api_key = key_manager.get_next_key().strip()
            if not api_key:
                st.error("❌ Không tìm thấy API Key nào trong key_manager.py!")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }

            response = requests.post(
                GROQ_API_URL, headers=headers, json=payload, timeout=60
            )
            res_data = response.json()

            if response.status_code == 200:
                try:
                    return res_data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    st.warning("⚠️ Nhận được phản hồi nhưng định dạng không đúng.")
                    return None

            err_msg = str(res_data).lower()

            # Model bị khai tử — không thể retry, báo lỗi rõ ngay
            if "model_decommissioned" in err_msg or "decommissioned" in err_msg:
                decommissioned_name = res_data.get("error", {}).get("message", model_name)
                st.error(
                    f"❌ Model bị Groq khai tử: `{model_name}`.\n"
                    f"Chi tiết: {decommissioned_name}\n"
                    "Đã tự động dùng model dự phòng ở lần gọi tiếp theo."
                )
                # Tự động fallback sang model còn lại
                if model_name == DEFAULT_MODEL:
                    model_name = MODEL_LITE
                elif model_name == MODEL_LITE:
                    model_name = DEFAULT_MODEL
                continue

            # Rate limit / quota — đổi key
            if any(code in err_msg for code in ["429", "rate_limit", "quota"]):
                st.toast("🔄 Key Groq bị giới hạn tốc độ, đang đổi Key...")
                time.sleep(2)
                continue

            if attempt == max_retries - 1:
                st.error(f"❌ Lỗi từ Groq API: {res_data}")
                return None
            time.sleep(2)

        except Exception as exc:
            if attempt == max_retries - 1:
                st.error(f"❌ Lỗi kết nối Groq: {exc}")
                return None
            time.sleep(2)

    return None


# ============================================================
# 2. XỬ LÝ TÀI LIỆU
# ============================================================

def extract_text_from_file(uploaded_file) -> str:
    import fitz
    import docx
    import pandas as pd

    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".pdf"):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
            return text
        elif file_name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            return "\n".join([p.text for p in doc.paragraphs])
        elif file_name.endswith((".xlsx", ".xls", ".csv")):
            df = (
                pd.read_csv(uploaded_file)
                if file_name.endswith(".csv")
                else pd.read_excel(uploaded_file)
            )
            return df.to_markdown()
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return f"Lỗi trích xuất: {exc}"
    finally:
        gc.collect()


# ============================================================
# 3. HỆ THỐNG WRITING PIPELINE
# ============================================================

def generate_evidence_based(
    task_prompt: str,
    evidence: List[Dict[str, Any]],
    citation_engine,
    study_context: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Any, List[str]]:

    if not evidence:
        return "Tài liệu chưa đủ bằng chứng.", evidence, []

    evidence_context = ""
    for ev in evidence:
        tag = citation_engine.register_evidence(ev["source_id"], ev.get("metadata", {}))
        evidence_context += f"\nTài liệu {tag}: {ev.get('text', '')}\n"

    context_str = ""
    if study_context and any(study_context.values()):
        context_str = "\nBỐI CẢNH ĐỀ TÀI:\n" + "\n".join(
            [f"- {k}: {v}" for k, v in study_context.items()]
        )

    # Bước 1: Lập dàn ý (dùng model nhẹ để tiết kiệm quota)
    outline_prompt = (
        f"{BASE_SYSTEM_RULES}\n{context_str}\n"
        f"NHIỆM VỤ: Lập dàn ý 3-4 luận điểm cho: {task_prompt}\n"
        f"BẰNG CHỨNG: {evidence_context}"
    )
    outline_res = call_gemini(outline_prompt, model=MODEL_LITE, temperature=0.1)
    structured_outline = outline_res if outline_res else "1. Đặt vấn đề\n2. Phân tích\n3. Kết luận"

    # Bước 2: Viết nháp đầy đủ (dùng model mạnh)
    draft_prompt = (
        f"{BASE_SYSTEM_RULES}\n{context_str}\n"
        f"DÀN Ý: {structured_outline}\n"
        f"NHIỆM VỤ: {task_prompt}\n"
        f"BẰNG CHỨNG: {evidence_context}"
    )
    raw_output = call_gemini(draft_prompt, temperature=0.0)

    if not raw_output:
        return None, evidence, []

    # Bước 3: Xử lý citation
    final_text, references, invalid_tags = citation_engine.process_vancouver_citations(raw_output)
    return final_text, references, invalid_tags
