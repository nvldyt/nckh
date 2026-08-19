import pandas as pd
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

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
    """Hàm xử lý độc lập cho 1 bảng (Dùng để chạy Đa luồng)"""
    table_title = decision.title
    table_markdown = df_table.to_markdown(index=False)
    
    ch3_part = f"## 3.{idx}. {table_title}\n\n{table_markdown}\n\n"
    ch4_part = ""

    # --- SUB-AGENT 1: TỰ ĐỘNG VIẾT NHẬN XÉT BẢNG ---
    remark_prompt = f"""
{BASE_SYSTEM_RULES}
Nhiệm vụ: Dựa vào bảng số liệu dưới đây, hãy viết phần 'Nhận xét' ngắn gọn, khoa học, CHỈ diễn giải các số liệu nổi bật (giá trị cao nhất, thấp nhất, tỷ lệ %). 
TUYỆT ĐỐI KHÔNG giải thích nguyên nhân, KHÔNG bàn luận. Viết thành MỘT đoạn văn xuôi liền mạch.
BẢNG SỐ LIỆU (Bảng {idx} - {table_title}):
{table_markdown}
"""
    # Nhiệt độ 0.0 để tuyệt đối trung thành với số liệu trong bảng
    table_remark = call_gemini(remark_prompt, temperature=0.0)
    if table_remark:
        ch3_part += f"**Nhận xét:** {table_remark}\n\n"

    # --- SUB-AGENT 2: RAG TÌM BẰNG CHỨNG ĐỐI CHIẾU ---
    # Tối ưu hóa câu truy vấn để Vector Embedding bắt ngữ nghĩa tốt hơn
    vars_str = ', '.join(decision.variables) if hasattr(decision, 'variables') else ''
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

    if disc_text:
        ch4_part = f"## 4.{idx}. Bàn luận về {table_title}\n\n{disc_text}\n\n"

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
    Loop Agent tự động lắp ráp Chương Kết quả và Bàn luận (Tích hợp Đa luồng siêu tốc).
    """
    if not selection_decisions or not saved_tables:
        return "Chưa có quyết định tuyển chọn bảng hoặc giỏ bảng trống.", ""

    # Sắp xếp các quyết định theo thứ tự khuyến nghị
    sorted_decisions = sorted(
        [d for d in selection_decisions if d.recommended_order],
        key=lambda x: int(x.recommended_order) if str(x.recommended_order).isdigit() else 999
    )

    # Khởi tạo sườn văn bản
    results_content = ["# CHƯƠNG 3: KẾT QUẢ NGHIÊN CỨU\n\n"]
    discussion_content = ["# CHƯƠNG 4: BÀN LUẬN\n\n"]
    
    # Dictionary để lưu kết quả theo đúng thứ tự idx
    ch3_results = {}
    ch4_results = {}

    # Chạy đa luồng (Max 4 workers để tránh bị Google API Rate Limit)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, decision in enumerate(sorted_decisions, start=1):
            if decision.result_id in saved_tables:
                df_table = saved_tables[decision.result_id]
                futures.append(
                    executor.submit(
                        process_single_decision, 
                        idx, decision, df_table, chunks, embeddings, bm25, citation_engine, study_context
                    )
                )

        # Gom kết quả khi các luồng hoàn thành
        for future in concurrent.futures.as_completed(futures):
            try:
                i, ch3_part, ch4_part = future.result()
                ch3_results[i] = ch3_part
                ch4_results[i] = ch4_part
            except Exception as e:
                print(f"Lỗi khi xử lý một bảng: {e}")

    # Ráp lại văn bản theo đúng thứ tự ban đầu
    for i in sorted(ch3_results.keys()):
        results_content.append(ch3_results[i])
        discussion_content.append(ch4_results[i])

    return "".join(results_content), "".join(discussion_content)
