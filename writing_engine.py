import os
import time
import gc
import io
import requests
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional

import key_manager 

# ============================================================
# CẤU HÌNH MODEL GROQ (Sử dụng Llama 3.3 70B viết y khoa cực đỉnh)
# ============================================================
DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============================================================
# 1. HÀM GỌI AI THÔNG QUA GROQ API
# ============================================================

def call_gemini(prompt: str, model: str = None, temperature: float = 0.3, max_retries: int = 3) -> str:
    """Hàm gọi AI sử dụng Groq API (giữ nguyên tên hàm để không ảnh hưởng các Tab khác)"""
    model_name = model if model else DEFAULT_MODEL
    
    for attempt in range(max_retries):
        try:
            api_key = key_manager.get_next_key().strip()
            if not api_key:
                st.error("❌ Không tìm thấy API Key nào trong hệ thống!")
                return None
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            res_data = response.json()
            
            if response.status_code == 200:
                try:
                    return res_data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    st.warning("⚠️ Nhận được phản hồi nhưng định dạng không đúng.")
                    return None
            else:
                err_msg = str(res_data).lower()
                if any(code in err_msg for code in ["429", "rate_limit", "quota"]):
                    st.toast(f"🔄 Key Groq bị giới hạn, đang đổi Key...")
                    time.sleep(2)
                    continue
                
                if attempt == max_retries - 1:
                    st.error(f"❌ Lỗi từ Groq API: {res_data}")
                    return None
                time.sleep(2)
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ Lỗi kết nối Groq: {e}")
                return None
            time.sleep(2)
            
    return None

# ============================================================
# 2. XỬ LÝ TÀI LIỆU
# ============================================================

def extract_text_from_file(uploaded_file):
    import fitz, docx, pandas as pd
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.pdf'):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
            return text
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([p.text for p in doc.paragraphs])
            return text
        elif file_name.endswith(('.xlsx', '.xls', '.csv')):
            df = pd.read_csv(uploaded_file) if file_name.endswith('.csv') else pd.read_excel(uploaded_file)
            return df.to_markdown()
        return uploaded_file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Lỗi trích xuất: {e}"
    finally:
        gc.collect()

# ============================================================
# 3. HỆ THỐNG PROMPT VÀ WRITING PIPELINE
# ============================================================

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

def generate_evidence_based(task_prompt, evidence, citation_engine, study_context=None):
    if not evidence:
        return "Tài liệu chưa đủ bằng chứng.", evidence, []

    evidence_context = ""
    for ev in evidence:
        tag = citation_engine.register_evidence(ev["source_id"], ev.get("metadata", {}))
        evidence_context += f"\nTài liệu {tag}: {ev.get('text', '')}\n"

    context_str = ""
    if study_context and any(study_context.values()):
        context_str = "\nBỐI CẢNH ĐỀ TÀI:\n" + "\n".join([f"- {k}: {v}" for k, v in study_context.items()])

    # Bước 1 & 2: Dàn ý
    outline_prompt = f"{BASE_SYSTEM_RULES}\n{context_str}\nNHIỆM VỤ: Lập dàn ý 3-4 luận điểm cho: {task_prompt}\nBẰNG CHỨNG: {evidence_context}"
    outline_res = call_gemini(outline_prompt, temperature=0.1)
    structured_outline = outline_res if outline_res else "1. Đặt vấn đề\n2. Phân tích\n3. Kết luận"

    # Bước 3: Viết nháp
    draft_prompt = f"{BASE_SYSTEM_RULES}\n{context_str}\nDÀN Ý: {structured_outline}\nNHIỆM VỤ: {task_prompt}\nBẰNG CHỨNG: {evidence_context}"
    raw_output = call_gemini(draft_prompt, temperature=0.0) 
    
    if not raw_output: return None, evidence, []

    # Bước 4: Citation
    final_text, references, invalid_tags = citation_engine.process_vancouver_citations(raw_output)
    return final_text, references, invalid_tags
