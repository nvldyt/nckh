import time
import pandas as pd
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

from writing_engine import call_gemini, generate_evidence_based, BASE_SYSTEM_RULES
from retrieval_engine import retrieve_evidence
from citation_engine import CitationEngine

def process_single_decision(
    idx: int,
    decision: Any,
    df_table: pd.DataFrame,
    chunks: List[Dict[str, Any]],
    embeddings: Any,
    bm25: Any,
    citation_engine: CitationEngine,
    study_context: Optional[Dict[str, Any]]
) -> Tuple[int, str, str]:
    """Hàm xử lý độc lập cho 1 bảng (Đã tối ưu chống nghẽn API và bắt lỗi an toàn)"""
    
    # --- XỬ LÝ AN TOÀN: Hỗ trợ cả Object lẫn Dictionary cho decision ---
    if isinstance(decision, dict):
        table_title = decision.get("title", f"Bảng {idx}")
        variables = decision.get("variables", [])
    else:
        table_title = getattr(decision, "title", f"Bảng {idx}")
        variables = getattr(decision, "variables", [])

    table_markdown = df_table.to_markdown(index=False)
    
    ch3_part = f"## 3.{idx}. {table_title}\n\n{table_markdown}\n\n"
    ch4_part = ""

    try:
        # --- SUB-AGENT 1: TỰ ĐỘNG VIẾT NHẬN XÉT BẢNG ---
        remark_prompt = f"""
{BASE_SYSTEM_RULES}
Nhiệm vụ: Dựa vào bảng số liệu dưới đây, hãy viết phần 'Nhận xét' ngắn gọn, khoa học, CHỈ diễn giải các số liệu nổi bật (giá trị cao nhất, thấp nhất, tỷ lệ %). 
TUYỆT ĐỐI KHÔNG giải thích nguyên nhân, KHÔNG bàn luận. Viết thành MỘT đoạn văn xuôi liền mạch.
BẢNG SỐ LIỆU (Bảng {idx} - {table_title}):
{table_markdown}
"""
        table_remark = call_gemini(remark_prompt, temperature=0.0)
        
        # Hãm phanh chống dội bom Google API Rate Limit
        time.sleep(2) 

        if table_remark:
            ch3_part += f"**Nhận xét:** {table_remark}\n\n"

        # --- SUB-AGENT 2: RAG TÌM BẰNG CHỨNG ĐỐI CHIẾU ---
        vars_str = ', '.join(variables) if variables else ''
        query_for_rag = f"Kết quả nghiên cứu và bàn luận về {table_title}. Mối liên quan của các yếu tố: {vars_str}."
        
        evidence = retrieve_evidence(query_for_rag, chunks, embeddings, bm25, top_k=6)

        # --- SUB-AGENT 3: TỰ ĐỘNG VIẾT BÀN LUẬN & SO SÁNH ---
        discussion_task = f"""
Dựa trên kết quả thực tế của bảng số liệu dưới đây, hãy viết phần BÀN LUẬN chuyên sâu cho luận văn CKI Dược lâm sàng:
- Tiêu đề bảng: {table_title}
- Số liệu tóm tắt: 
{table_markdown}
- Nhận xét số liệu: {table_remark if table_remark else 'Không có nhận xét chi tiết'}

YÊU CẦU BẮT BUỘC:
1. Giải thích nguyên nhân, cơ sở lâm sàng hoặc dược lý dẫn đến kết quả này.
2. Đối chiếu, so sánh trực tiếp kết quả với các tài liệu y văn được cung cấp.
3. Viết thành các đoạn văn xuôi y khoa liền mạch, khách quan.
4. CHỈ SỬ DỤNG bằng chứng được cung cấp, không tự bịa số liệu.
"""
        disc_text, _, _ = generate_evidence_based(
            task_prompt=discussion_task,
            evidence=evidence,
            citation_engine=citation_engine,
            study_context=study_context
        )
        
        # Thêm nhịp nghỉ an toàn cho luồng
        time.sleep(2)

        if disc_text:
            ch4_part = f"## 4.{idx}. Bàn luận về {table_title}\n\n{disc_text}\n\n"

    except Exception as exc:
        # Hiển thị lỗi đỏ chi tiết lên giao diện Streamlit thay vì nuốt mất lỗi
        st.error(f"Lỗi khi xử lý bảng [{table_title}]: {exc}")
        
    return idx, ch3_part, ch4_part

def assemble_results_and_discussion_chapter(
    selection_decisions: List[Any],
    saved_tables: Dict[str, pd.DataFrame],
    chunks: List[Dict[str, Any]],
    embeddings: Any,
    bm25: Any,
    citation_engine: CitationEngine,
    study_context: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Loop Agent tự động lắp ráp Chương Kết quả và Bàn luận 
    (Tích hợp cơ chế Fallback chống rỗng và Đa luồng kiểm soát tốc độ).
    """
    if not saved_tables:
        return "⚠️ Giỏ kết quả đang trống, không có bảng nào để tổng hợp.", ""

    # --- BƯỚC PHÒNG THỦ: Tránh bị lọc sạch nếu selection_decisions rỗng hoặc thiếu recommended_order ---
    valid_decisions = []
    if selection_decisions:
        for d in selection_decisions:
            rid = getattr(d, 'result_id', None) or (d.get('result_id') if isinstance(d, dict) else None)
            if rid and rid in saved_tables:
                valid_decisions.append(d)

    # Nếu bộ lọc quét không ra kết quả, tự động kích hoạt CƠ CHẾ CỨU NGUY (Lấy toàn bộ bảng trong Giỏ)
    if not valid_decisions:
        st.warning("⚠️ Phát hiện danh sách tuyển chọn trống. Hệ thống đang tự động kích hoạt chế độ lấy toàn bộ bảng trong Giỏ kết quả để lập bản thảo.")
        for tid in saved_tables.keys():
            class FallbackDecision:
                def __init__(self, tid):
                    self.result_id = tid
                    self.title = tid
                    self.variables = []
                    self.recommended_order = 1
            valid_decisions.append(FallbackDecision(tid))

    # Sắp xếp các quyết định theo thứ tự khuyến nghị an toàn
    try:
        sorted_decisions = sorted(
            valid_decisions,
            key=lambda x: int(getattr(x, 'recommended_order', 999) if not isinstance(x, dict) else x.get('recommended_order', 999)) 
            if str(getattr(x, 'recommended_order', 999) if not isinstance(x, dict) else x.get('recommended_order', 999)).isdigit() else 999
        )
    except Exception:
        sorted_decisions = valid_decisions

    # Khởi tạo sườn văn bản hai chương
    results_content = ["# CHƯƠNG 3: KẾT QUẢ NGHIÊN CỨU\n\n"]
    discussion_content = ["# CHƯƠNG 4: BÀN LUẬN\n\n"]
    
    ch3_results = {}
    ch4_results = {}

    # Chạy đa luồng an toàn (Giới hạn max_workers=2 để chống lỗi 429 - Too Many Requests từ Google API)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, decision in enumerate(sorted_decisions, start=1):
            rid = getattr(decision, 'result_id', None) or (decision.get('result_id') if isinstance(decision, dict) else "")
            if rid in saved_tables:
                df_table = saved_tables[rid]
                futures.append(
                    executor.submit(
                        process_single_decision, 
                        idx, decision, df_table, chunks, embeddings, bm25, citation_engine, study_context
                    )
                )

        # Thu gom kết quả từ các luồng
        for future in concurrent.futures.as_completed(futures):
            try:
                i, ch3_part, ch4_part = future.result()
                if ch3_part:
                    ch3_results[i] = ch3_part
                if ch4_part:
                    ch4_results[i] = ch4_part
            except Exception as e:
                st.error(f"Lỗi luồng đồng thời: {e}")

    # Ráp lại văn bản theo đúng thứ tự chỉ mục (idx) ban đầu
    for i in sorted(ch3_results.keys()):
        results_content.append(ch3_results[i])
    for i in sorted(ch4_results.keys()):
        discussion_content.append(ch4_results[i])

    # Kiểm tra lần cuối nếu nội dung quá ngắn
    final_ch3 = "".join(results_content)
    final_ch4 = "".join(discussion_content)

    if len(final_ch3.strip()) <= len("# CHƯƠNG 3: KẾT QUẢ NGHIÊN CỨU\n\n"):
        return "⚠️ Không thể tạo nội dung Chương 3. Vui lòng kiểm tra lại phản hồi từ API Gemini.", ""

    return final_ch3, final_ch4
