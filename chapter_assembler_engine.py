# chapter_assembler_engine.py
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from writing_engine import call_gemini, generate_evidence_based, BASE_SYSTEM_RULES
from retrieval_engine import retrieve_evidence
from citation_engine import CitationEngine

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
    Loop Agent tự động lắp ráp Chương Kết quả và Chương Bàn luận dựa trên mạch kể chuyện.
    """
    if not selection_decisions or not saved_tables:
        return "Chưa có quyết định tuyển chọn bảng hoặc giỏ bảng trống.", ""

    results_chapter_content = "# CHƯƠNG 3: KẾT QUẢ NGHIÊN CỨU\n\n"
    discussion_chapter_content = "# CHƯƠNG 4: BÀN LUẬN\n\n"

    # Sắp xếp các quyết định theo thứ tự khuyến nghị của Table Selection Engine
    sorted_decisions = sorted(
        [d for d in selection_decisions if d.recommended_order],
        key=lambda x: int(x.recommended_order) if str(x.recommended_order).isdigit() else 999
    )

    for idx, decision in enumerate(sorted_decisions, start=1):
        table_id = decision.result_id
        if table_id not in saved_tables:
            continue
        
        df_table = saved_tables[table_id]
        table_title = decision.title
        table_markdown = df_table.to_markdown(index=False)

        # Đưa bảng vào Chương 3
        results_chapter_content += f"## 3.{idx}. {table_title}\n\n"
        results_chapter_content += f"{table_markdown}\n\n"

        # --- SUB-AGENT 1: TỰ ĐỘNG VIẾT NHẬN XÉT BẢNG ---
        remark_prompt = f"""
{BASE_SYSTEM_RULES}
Nhiệm vụ: Dựa vào bảng số liệu dưới đây, hãy viết phần 'Nhận xét' ngắn gọn, khoa học, CHỈ diễn giải các số liệu nổi bật (giá trị cao nhất, thấp nhất, tỷ lệ %). 
TUYỆT ĐỐI KHÔNG giải thích nguyên nhân, KHÔNG bàn luận. Viết thành một đoạn văn xuôi liền mạch.
BẢNG SỐ LIỆU (Bảng {idx} - {table_title}):
{table_markdown}
"""
        table_remark = call_gemini(remark_prompt, temperature=0.1)
        if table_remark:
            results_chapter_content += f"**Nhận xét:** {table_remark}\n\n"

        # --- SUB-AGENT 2: RAG TÌM BẰNG CHỨNG ĐỐI CHIẾU CHO BÀN LUẬN ---
        query_for_rag = f"{table_title} {' '.join(decision.variables)}"
        evidence = retrieve_evidence(query_for_rag, chunks, embeddings, bm25, top_k=6)

        # --- SUB-AGENT 3: TỰ ĐỘNG VIẾT BÀN LUẬN & SO SÁNH ---
        discussion_task = f"""
Dựa trên kết quả thực tế của bảng số liệu dưới đây, hãy viết phần BÀN LUẬN chuyên sâu cho luận văn CKI Dược lâm sàng:
- Tiêu đề bảng: {table_title}
- Số liệu tóm tắt: 
{table_markdown}
- Nhận xét số liệu: {table_remark if table_remark else 'Không có nhận xét chi tiết'}

YÊU CẦU:
1. Giải thích nguyên nhân, cơ sở lâm sàng hoặc dược lý dẫn đến kết quả này.
2. Đối chiếu, so sánh trực tiếp kết quả với các tài liệu y văn được cung cấp trong phần bằng chứng.
3. Viết thành các đoạn văn xuôi y khoa liền mạch, tuyệt đối không dùng gạch đầu dòng ngắt vụn.
4. Chèn mã trích dẫn [REF-...] ở cuối câu khi lấy thông tin từ tài liệu.
"""
        disc_text, _, _ = generate_evidence_based(
            task_prompt=discussion_task,
            evidence=evidence,
            citation_engine=citation_engine,
            study_context=study_context
        )

        if disc_text:
            discussion_chapter_content += f"## 4.{idx}. Bàn luận về {table_title}\n\n"
            discussion_chapter_content += f"{disc_text}\n\n"

    return results_chapter_content, discussion_chapter_content
