# reference_engine.py
from typing import Dict, Any, List

def format_vancouver_citation(meta: Dict[str, Any], vancouver_index: int) -> str:
    """Ép chuẩn định dạng Vancouver Dược lâm sàng."""
    authors = meta.get('authors', '').strip()
    title = meta.get('title', '').strip()
    journal = meta.get('journal', '').strip()
    year = meta.get('year', '').strip()
    
    # Xử lý nếu thiếu dữ liệu
    if not authors: authors = "[Thiếu tên tác giả]"
    if not title: title = meta.get('file_name', '[Thiếu tên bài báo]')
    if not journal: journal = "[Thiếu tên tạp chí]"
    if not year: year = "[Năm]"

    # Chuẩn Vancouver: Tác giả. Tên bài. Tên tạp chí. Năm.
    citation = f"[{vancouver_index}] {authors}. {title}. {journal}. {year}."
    
    # Bổ sung DOI/URL nếu có
    if meta.get("doi"): 
        citation += f" DOI: {meta['doi']}."
    elif meta.get("url"): 
        citation += f" Có tại: {meta['url']}"
        
    return citation

def build_bibliography(references: List[Dict[str, Any]], documents: Dict[str, Any]) -> str:
    """Xây dựng danh mục TLTK từ danh sách các citation đã dùng trong bản nháp."""
    rows = []
    for ref in references:
        source_id = ref['ref_id'].replace("REF-", "") if ref['ref_id'].startswith("REF-") else ref['ref_id']
        meta = documents.get(source_id, ref.get("metadata", {}))
        
        vancouver_str = format_vancouver_citation(meta, ref['vancouver_index'])
        rows.append(vancouver_str)
        
    return "\n".join(rows)
