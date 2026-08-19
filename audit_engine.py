import re
import math
from typing import List, Dict, Any, Optional

# ============================================================
# 1. KIỂM ĐỊNH SỐ LIỆU (NUMERIC AUDIT)
# ============================================================

def extract_numeric_tokens(text: str) -> List[str]:
    """Trích xuất các con số từ văn bản (bao gồm số thập phân và %)."""
    if not text:
        return []
    pattern = r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*%)?"
    return re.findall(pattern, text)

def parse_number(num_str: str) -> Optional[float]:
    """Chuyển chuỗi số thành dạng float để tính toán."""
    clean_str = num_str.replace(",", ".").replace(" ", "").replace("%", "")
    try:
        return float(clean_str)
    except ValueError:
        return None

def _strip_citation_markers(text: str) -> str:
    """
    Loại bỏ các mã trích dẫn dạng [1], [2]... hoặc REF-xxx 
    để thuật toán không nhận nhầm số thứ tự tài liệu thành số liệu thống kê.
    """
    if not text:
        return text
    cleaned = re.sub(r"\[\s*\d+\s*\]", " ", text)
    cleaned = re.sub(r"REF-[A-Za-z0-9\-]+", " ", cleaned)
    return cleaned

def compare_numbers_advanced(source_text: str, generated_text: str) -> Dict[str, Any]:
    """
    So sánh số liệu giữa bài viết của AI và văn bản gốc.
    Phân loại thành: Khớp chính xác, Khớp phái sinh (toán học), và Số liệu lạ.
    """
    generated_text_clean = _strip_citation_markers(generated_text)

    source_nums = extract_numeric_tokens(source_text)
    generated_nums = extract_numeric_tokens(generated_text_clean)

    # Chuẩn hóa format số
    source_normalized = set(x.replace(",", ".").replace(" ", "") for x in source_nums)
    generated_normalized = set(x.replace(",", ".").replace(" ", "") for x in generated_nums)
    source_floats = set(filter(None, [parse_number(x) for x in source_normalized]))

    exact_matches = []
    derived_matches = []
    warnings = []

    for gen_num in generated_normalized:
        # Level 1: Khớp chính xác hoàn toàn 1-1
        if gen_num in source_normalized:
            exact_matches.append(gen_num)
        else:
            gen_val = parse_number(gen_num)
            if gen_val is not None:
                is_derived = False
                # Level 2: Khớp phái sinh (Ví dụ: 0.5 == 50%)
                for src_val in source_floats:
                    if math.isclose(gen_val, src_val, rel_tol=1e-4) or \
                       math.isclose(gen_val, src_val * 100, rel_tol=1e-4) or \
                       math.isclose(gen_val * 100, src_val, rel_tol=1e-4):
                        is_derived = True
                        break

                if is_derived:
                    derived_matches.append(gen_num)
                else:
                    warnings.append(gen_num)
            else:
                warnings.append(gen_num)

    return {
        "exact_matches": sorted(exact_matches),
        "derived_matches": sorted(derived_matches),
        "warnings": sorted(warnings),
        "source_raw": sorted(source_normalized)
    }

def Audit_generated_text(text: str, relevant_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hàm tổng hợp để kiểm định một đoạn văn bản.
    Nhận vào đoạn văn của AI và danh sách các chunks bằng chứng để đối chiếu.
    """
    source_text = "\n".join(e["text"] for e in relevant_evidence)
    audit_result = compare_numbers_advanced(source_text, text)
    return {
        "evidence_used": relevant_evidence,
        **audit_result
    }

# ============================================================
# 2. KIỂM ĐỊNH TRÙNG LẶP / ĐẠO VĂN (PLAGIARISM & OVERLAP)
# ============================================================

def normalize_for_similarity(text: str) -> str:
    """Làm sạch văn bản để so sánh Jaccard."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.,-]", "", text)
    return text.strip()

def ngram_set(text: str, n: int = 8) -> set:
    """Cắt văn bản thành các N-grams để tìm chuỗi trùng lặp."""
    words = normalize_for_similarity(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

def internal_overlap_Audit(text: str, all_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Quét xem đoạn văn AI viết có bị copy-paste y nguyên (trùng lặp) 
    từ một đoạn nào đó trong kho tài liệu hay không.
    """
    target = ngram_set(text)
    if not target:
        return []

    results = []
    for chunk in all_chunks:
        other = ngram_set(chunk["text"])
        if not other:
            continue

        intersection = len(target & other)
        union = len(target | other)
        if union == 0:
            continue

        jaccard = intersection / union
        if jaccard > 0:
            results.append({
                "file": chunk.get("file_name", "Unknown"), 
                "page": chunk.get("page", "Unknown"),
                "chunk_id": chunk.get("chunk_id", "Unknown"), 
                "similarity": round(jaccard, 4),
                "text": chunk.get("text", "")[:500],
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
