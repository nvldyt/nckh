import re
import math
import concurrent.futures
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Tuple

# ============================================================
# 1. MÁY HỌC SUY LUẬN SỐ LIỆU (NUMERIC REASONING ENGINE)
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
    """Loại bỏ các mã trích dẫn để thuật toán không nhận nhầm thành số liệu thống kê."""
    if not text:
        return text
    cleaned = re.sub(r"\[\s*\d+\s*\]", " ", text)
    cleaned = re.sub(r"REF-[A-Za-z0-9\-]+", " ", cleaned)
    return cleaned

def check_numeric_relationship(gen_val: float, source_floats: set) -> Tuple[bool, str]:
    """
    Suy luận toán học: Tối ưu thêm abs_tol=1e-5 để chống lỗi với số 0.
    """
    src_list = list(source_floats)
    n = len(src_list)
    
    for i in range(n):
        for j in range(n):
            if i == j or src_list[j] == 0:
                continue
            a, b = src_list[i], src_list[j]
            
            if math.isclose(a / b, gen_val, rel_tol=1e-3, abs_tol=1e-5):
                return True, f"{a} / {b} = {gen_val}"
            if math.isclose((a / b) * 100, gen_val, rel_tol=1e-3, abs_tol=1e-5):
                return True, f"({a} / {b}) * 100 = {gen_val}%"
            
    for i in range(n):
        for j in range(i+1, n):
            a, b = src_list[i], src_list[j]
            if math.isclose(a + b, gen_val, rel_tol=1e-3, abs_tol=1e-5):
                return True, f"{a} + {b} = {gen_val}"
            if math.isclose(abs(a - b), gen_val, rel_tol=1e-3, abs_tol=1e-5):
                return True, f"|{a} - {b}| = {gen_val}"
            
    return False, ""

def compare_numbers_advanced(source_text: str, generated_text: str) -> Dict[str, Any]:
    """
    So sánh số liệu với 3 cấp độ: Khớp chính xác -> Khớp nội suy/Toán học -> Số liệu lạ.
    """
    generated_text_clean = _strip_citation_markers(generated_text)

    source_nums = extract_numeric_tokens(source_text)
    generated_nums = extract_numeric_tokens(generated_text_clean)

    source_normalized = set(x.replace(",", ".").replace(" ", "") for x in source_nums)
    generated_normalized = set(x.replace(",", ".").replace(" ", "") for x in generated_nums)
    source_floats = set(filter(None, [parse_number(x) for x in source_normalized]))

    exact_matches = []
    derived_matches = []
    warnings = []
    reasoning_logs = []

    for gen_num in generated_normalized:
        if gen_num in source_normalized:
            exact_matches.append(gen_num)
        else:
            gen_val = parse_number(gen_num)
            if gen_val is not None:
                is_derived = False
                for src_val in source_floats:
                    if math.isclose(gen_val, src_val, rel_tol=1e-4, abs_tol=1e-5) or \
                       math.isclose(gen_val, src_val * 100, rel_tol=1e-4, abs_tol=1e-5) or \
                       math.isclose(gen_val * 100, src_val, rel_tol=1e-4, abs_tol=1e-5):
                        is_derived = True
                        derived_matches.append(gen_num)
                        reasoning_logs.append(f"Số [{gen_num}] được quy đổi định dạng từ ({src_val})")
                        break
                
                if not is_derived:
                    is_math, math_log = check_numeric_relationship(gen_val, source_floats)
                    if is_math:
                        derived_matches.append(gen_num)
                        reasoning_logs.append(f"Số [{gen_num}] hợp lệ do AI tự tính: {math_log}")
                    else:
                        warnings.append(gen_num)
            else:
                warnings.append(gen_num)

    return {
        "exact_matches": sorted(exact_matches),
        "derived_matches": sorted(derived_matches),
        "warnings": sorted(warnings),
        "reasoning_logs": reasoning_logs,
        "source_raw": sorted(source_normalized)
    }

# ============================================================
# 2. KIỂM ĐỊNH TỪNG LUẬN ĐIỂM (CLAIM-LEVEL AUDIT)
# ============================================================

def split_into_claims(text: str) -> List[str]:
    """Chẻ đoạn văn dài thành từng câu luận điểm độc lập."""
    claims = re.split(r'(?<=[.!?])\s+', text.strip())
    return [c.strip() for c in claims if len(c.strip()) > 15]

def audit_single_claim(claim: str, retriever_func: Callable, top_k: int) -> Dict[str, Any]:
    """Hàm phụ trợ để xử lý 1 câu luận điểm (chạy đa luồng)."""
    evidence = retriever_func(claim, top_k)
    source_text = "\n".join(e["text"] for e in evidence)
    num_audit = compare_numbers_advanced(source_text, claim)
    
    citations_in_claim = re.findall(r"\[(REF-[a-zA-Z0-9_-]+)\]", claim)
    citations_in_claim += re.findall(r"\[(\d+)\]", claim)
    
    return {
        "claim": claim,
        "evidence_used": evidence,
        "citations_found": citations_in_claim,
        "numeric_audit": num_audit
    }

def claim_level_audit(
    generated_text: str, 
    retriever_func: Callable[[str, int], List[Dict[str, Any]]], 
    top_k_evidence: int = 4
) -> List[Dict[str, Any]]:
    """
    Quy trình Audit theo chuẩn Agentic (Đã tích hợp Đa luồng siêu tốc).
    """
    claims = split_into_claims(generated_text)
    audit_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(audit_single_claim, claim, retriever_func, top_k_evidence): claim for claim in claims}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                audit_results.append(future.result())
            except Exception:
                pass
                
    return audit_results

def Audit_generated_text(text: str, relevant_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hàm tổng hợp để kiểm định toàn bộ đoạn văn bản (Dùng cho giao diện hiện tại).
    """
    source_text = "\n".join(e["text"] for e in relevant_evidence)
    audit_result = compare_numbers_advanced(source_text, text)
    return {
        "evidence_used": relevant_evidence,
        **audit_result
    }

# ============================================================
# 3. KIỂM ĐỊNH TRÙNG LẶP / ĐẠO VĂN (PLAGIARISM & OVERLAP)
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
    Quét trùng lặp với cơ chế Cache N-grams tự động (Tránh tính lại cho Chunk cũ).
    """
    target = ngram_set(text)
    if not target:
        return []

    results = []
    for chunk in all_chunks:
        if "_cached_ngrams" not in chunk:
            chunk["_cached_ngrams"] = ngram_set(chunk.get("text", ""))
            
        other = chunk["_cached_ngrams"]
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

# ============================================================
# 4. KIỂM ĐỊNH TÍNH NHẤT QUÁN CỦA BẢNG (TABLE CONSISTENCY VALIDATOR)
# ============================================================

def auto_validate_table_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tự động kiểm tra tính nhất quán của bảng số liệu (DataFrame).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"is_consistent": True, "messages": ["Bảng dữ liệu trống hoặc không hợp lệ."]}

    messages = []
    is_consistent = True

    total_row_idx = -1
    total_keywords = ["tổng", "total", "chung", "cộng", "overall", "n"]
    
    for idx, row in df.iterrows():
        first_val = str(row.iloc[0]).strip().lower()
        if any(kw in first_val for kw in total_keywords):
            total_row_idx = idx
            break

    if total_row_idx == -1:
        return {"is_consistent": True, "messages": ["ℹ️ Không tìm thấy dòng Tổng/Total để đối chiếu tự động."]}

    data_df = df.drop(index=total_row_idx)
    total_row = df.loc[total_row_idx]

    for col in df.columns[1:]:
        try:
            cleaned_col = data_df[col].astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.')
            numeric_vals = pd.to_numeric(cleaned_col, errors='coerce')
            
            reported_total_raw = str(total_row[col])
            reported_total = parse_number(reported_total_raw)

            if numeric_vals.notna().any() and reported_total is not None:
                calculated_sum = numeric_vals.sum()
                if not math.isclose(calculated_sum, reported_total, rel_tol=1e-2):
                    is_consistent = False
                    messages.append(
                        f"⚠️ Lệch tổng ở cột '{col}': Các dòng cộng lại = {calculated_sum:,.2f}, nhưng bảng ghi tổng = {reported_total:,.2f}"
                    )
        except Exception:
            continue

    if is_consistent:
        messages.append("✅ Bảng số liệu hoàn toàn nhất quán (Cộng tổng các cột khớp chính xác).")

    return {
        "is_consistent": is_consistent,
        "messages": messages
    }
