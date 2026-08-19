# audit_engine.py
import re
import math
import concurrent.futures
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

import concurrent.futures

# [Giữ nguyên phần đầu...]

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
            
            # Thêm abs_tol để an toàn với các số gần 0
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

# [Giữ nguyên hàm compare_numbers_advanced và split_into_claims...]

# ============================================================
# TỐI ƯU ĐA LUỒNG: KIỂM ĐỊNH TỪNG LUẬN ĐIỂM
# ============================================================
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
            except Exception as e:
                # Xử lý an toàn nếu 1 luồng bị lỗi
                pass
                
    return audit_results

# ============================================================
# TỐI ƯU Caching JACCARD: KIỂM ĐỊNH ĐẠO VĂN
# ============================================================
def internal_overlap_Audit(text: str, all_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Quét trùng lặp với cơ chế Cache N-grams tự động (Tránh tính lại cho Chunk cũ).
    """
    target = ngram_set(text)
    if not target:
        return []

    results = []
    for chunk in all_chunks:
        # Lưu cache N-grams thẳng vào dict của chunk để dùng lại cho các lần Audit sau
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

# [Giữ nguyên phần validate_table_consistency...]
# ============================================================
# 4. KIỂM ĐỊNH TÍNH NHẤT QUÁN CỦA BẢNG (TABLE CONSISTENCY VALIDATOR)
# ============================================================

def auto_validate_table_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tự động kiểm tra tính nhất quán của bảng số liệu (DataFrame).
    Tìm dòng chứa từ khóa "Tổng", "Total", "Chung"... và kiểm tra xem 
    tổng các dòng bên trên có khớp với số liệu tổng được ghi hay không.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"is_consistent": True, "messages": ["Bảng dữ liệu trống hoặc không hợp lệ."]}

    messages = []
    is_consistent = True

    # Tìm dòng tổng (nếu có)
    total_row_idx = -1
    total_keywords = ["tổng", "total", "chung", "cộng", "overall", "n"]
    
    for idx, row in df.iterrows():
        first_val = str(row.iloc[0]).strip().lower()
        if any(kw in first_val for kw in total_keywords):
            total_row_idx = idx
            break

    if total_row_idx == -1:
        return {"is_consistent": True, "messages": ["ℹ️ Không tìm thấy dòng Tổng/Total để đối chiếu tự động."]}

    # Tách phần dữ liệu và dòng tổng
    data_df = df.drop(index=total_row_idx)
    total_row = df.loc[total_row_idx]

    # Kiểm tra từng cột có dữ liệu số (bỏ qua cột đầu tiên chứa nhãn)
    for col in df.columns[1:]:
        try:
            # Làm sạch chuỗi số để tính toán (loại bỏ dấu phẩy, khoảng trắng, ký tự phụ)
            cleaned_col = data_df[col].astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', '.')
            numeric_vals = pd.to_numeric(cleaned_col, errors='coerce')
            
            reported_total_raw = str(total_row[col])
            reported_total = parse_number(reported_total_raw)

            if numeric_vals.notna().any() and reported_total is not None:
                calculated_sum = numeric_vals.sum()
                # Sai số cho phép nhỏ (rel_tol = 1%)
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
