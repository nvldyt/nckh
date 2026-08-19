# audit_engine.py
import re
import math
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
    Suy luận toán học: Kiểm tra xem số liệu do AI sinh ra (gen_val) có phải là kết quả 
    của các phép toán (Tỷ lệ %, Cộng, Trừ) từ các số liệu gốc không.
    """
    src_list = list(source_floats)
    n = len(src_list)
    
    # 1. Kiểm tra phép chia (Tính Tỷ lệ / Tỷ lệ phần trăm)
    for i in range(n):
        for j in range(n):
            if i == j or src_list[j] == 0:
                continue
            a, b = src_list[i], src_list[j]
            
            # Khớp tỷ lệ nguyên (VD: 90/150 = 0.6)
            if math.isclose(a / b, gen_val, rel_tol=1e-3):
                return True, f"{a} / {b} = {gen_val}"
            # Khớp tỷ lệ phần trăm (VD: (90/150)*100 = 60%)
            if math.isclose((a / b) * 100, gen_val, rel_tol=1e-3):
                return True, f"({a} / {b}) * 100 = {gen_val}%"
            
    # 2. Kiểm tra phép cộng / trừ (Tính Tổng, phần bù)
    for i in range(n):
        for j in range(i+1, n):
            a, b = src_list[i], src_list[j]
            if math.isclose(a + b, gen_val, rel_tol=1e-3):
                return True, f"{a} + {b} = {gen_val}"
            if math.isclose(abs(a - b), gen_val, rel_tol=1e-3):
                return True, f"|{a} - {b}| = {gen_val}"
            
    return False, ""

def compare_numbers_advanced(source_text: str, generated_text: str) -> Dict[str, Any]:
    """
    So sánh số liệu với 3 cấp độ: Khớp chính xác -> Khớp nội suy/Toán học -> Số liệu lạ.
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
    reasoning_logs = [] # Lịch sử giải thích toán học

    for gen_num in generated_normalized:
        # Level 1: Khớp chính xác hoàn toàn 1-1
        if gen_num in source_normalized:
            exact_matches.append(gen_num)
        else:
            gen_val = parse_number(gen_num)
            if gen_val is not None:
                # Level 2A: Kiểm tra khớp định dạng thập phân (0.5 == 50)
                is_derived = False
                for src_val in source_floats:
                    if math.isclose(gen_val, src_val, rel_tol=1e-4) or \
                       math.isclose(gen_val, src_val * 100, rel_tol=1e-4) or \
                       math.isclose(gen_val * 100, src_val, rel_tol=1e-4):
                        is_derived = True
                        derived_matches.append(gen_num)
                        reasoning_logs.append(f"Số [{gen_num}] được quy đổi định dạng từ ({src_val})")
                        break
                
                # Level 2B: Máy học suy luận toán học (Numeric Reasoning)
                if not is_derived:
                    is_math, math_log = check_numeric_relationship(gen_val, source_floats)
                    if is_math:
                        derived_matches.append(gen_num)
                        reasoning_logs.append(f"Số [{gen_num}] hợp lệ do AI tự tính: {math_log}")
                    else:
                        warnings.append(gen_num)
            else:
                # Level 3: Báo động Số liệu lạ
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
    # Tách câu dựa trên dấu chấm, hỏi, than, theo sau là khoảng trắng
    claims = re.split(r'(?<=[.!?])\s+', text.strip())
    return [c.strip() for c in claims if len(c.strip()) > 15]

def claim_level_audit(
    generated_text: str, 
    retriever_func: Callable[[str, int], List[Dict[str, Any]]], 
    top_k_evidence: int = 4
) -> List[Dict[str, Any]]:
    """
    Quy trình Audit theo chuẩn Agentic: Soi từng câu, lấy bằng chứng riêng rẽ.
    """
    claims = split_into_claims(generated_text)
    audit_results = []
    
    for claim in claims:
        # Gọi RAG tìm bằng chứng MỚI chỉ tập trung hỗ trợ cho riêng LUẬN ĐIỂM NÀY
        evidence = retriever_func(claim, top_k_evidence)
        source_text = "\n".join(e["text"] for e in evidence)
        
        # Kiểm tra toán học trong phạm vi bằng chứng hẹp này
        num_audit = compare_numbers_advanced(source_text, claim)
        
        # Trích xuất mã Citation đang có trong câu
        citations_in_claim = re.findall(r"\[(REF-[a-zA-Z0-9_-]+)\]", claim)
        citations_in_claim += re.findall(r"\[(\d+)\]", claim)
        
        audit_results.append({
            "claim": claim,
            "evidence_used": evidence,
            "citations_found": citations_in_claim,
            "numeric_audit": num_audit
        })
        
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
