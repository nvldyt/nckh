# evidence_validator.py
# ============================================================
# BỘ LỌC VÀ CHẤM ĐIỂM BẰNG CHỨNG (EVIDENCE VALIDATOR)
# ============================================================
# Chức năng: Đánh giá chất lượng đa chiều của các đoạn bằng chứng 
# trước khi đưa vào LLM. Giúp loại bỏ "rác", ưu tiên nguồn uy tín,
# và chống trùng lặp nội dung.

import re
from datetime import datetime
from typing import List, Dict, Any

# Trọng số chấm điểm (Tổng = 1.0)
WEIGHT_SEMANTIC = 0.35      # Độ khớp ngữ nghĩa (Context)
WEIGHT_KEYWORD = 0.25       # Độ khớp từ khóa (BM25)
WEIGHT_SOURCE = 0.30        # Uy tín của nguồn (PubMed > VN Journal > PDF)
WEIGHT_COMPLETENESS = 0.10  # Mật độ thông tin của đoạn trích

MIN_VALID_SCORE = 0.40      # Ngưỡng điểm tối thiểu để được giữ lại
JACCARD_THRESHOLD = 0.75    # Ngưỡng loại bỏ trùng lặp (trùng >75% -> loại đoạn điểm thấp)

class EvidenceValidator:
    def __init__(self, current_year: int = None):
        self.current_year = current_year or datetime.now().year

    def _score_source_quality(self, meta: Dict[str, Any]) -> float:
        """
        Đánh giá uy tín của nguồn gốc bài báo (Tối đa 1.0).
        - Có DOI / PMID: +0.4
        - Nguồn PubMed: +0.3
        - Nguồn Tạp chí VN: +0.2
        - Năm xuất bản (Càng mới càng điểm cao): Tối đa +0.3
        """
        score = 0.0
        
        # 1. Điểm định danh quốc tế (0.4)
        if meta.get("doi") or meta.get("pmid"):
            score += 0.4
            
        # 2. Điểm xuất xứ (0.3)
        origin = meta.get("origin", "")
        if origin == "PubMed":
            score += 0.3
        elif origin == "Tạp chí VN":
            score += 0.2
        elif origin == "PDF":
            score += 0.15 # PDF thủ công, độ tin cậy phụ thuộc người dùng
            
        # 3. Điểm tính cập nhật (0.3)
        year_str = meta.get("year", "")
        try:
            # Tìm 4 chữ số liên tiếp đại diện cho năm
            match = re.search(r'\d{4}', str(year_str))
            if match:
                pub_year = int(match.group())
                age = max(0, self.current_year - pub_year)
                if age <= 3:
                    score += 0.3  # Nghiên cứu mới (<= 3 năm)
                elif age <= 5:
                    score += 0.2  # <= 5 năm
                elif age <= 10:
                    score += 0.1  # <= 10 năm
        except:
            pass # Không lấy được năm thì không cộng điểm
            
        return min(score, 1.0)

    def _score_completeness(self, text: str) -> float:
        """
        Đánh giá tính trọn vẹn của đoạn trích (Tối đa 1.0).
        Đoạn quá ngắn thường thiếu context, đoạn đủ dài chứa nhiều thông tin lâm sàng hơn.
        """
        word_count = len(text.split())
        if word_count < 20:
            return 0.2  # Quá ngắn, có thể là rác
        elif word_count < 50:
            return 0.5
        elif word_count < 100:
            return 0.8
        else:
            return 1.0  # Tối ưu cho LLM đọc hiểu

    def _calculate_jaccard(self, text1: str, text2: str) -> float:
        """Tính độ giao thoa từ vựng giữa 2 đoạn văn (chống trùng lặp)."""
        set1 = set(re.findall(r'\w+', text1.lower()))
        set2 = set(re.findall(r'\w+', text2.lower()))
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def validate_and_rank(
        self, 
        candidate_chunks: List[Dict[str, Any]], 
        documents_meta: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Quy trình lõi: Chấm điểm -> Lọc ngưỡng -> Lọc trùng lặp -> Xếp hạng.
        """
        if not candidate_chunks:
            return []

        validated_candidates = []

        # BƯỚC 1: CHẤM ĐIỂM ĐA CHIỀU
        for chunk in candidate_chunks:
            meta = documents_meta.get(chunk["source_id"], {})
            
            # Lấy điểm thô từ bước Retrieval (Giả định Hybrid Engine đã chuẩn hóa về 0-1)
            # Nếu hệ thống truyền vào điểm gộp (score), ta dùng nó cho cả hai, 
            # hoặc lý tưởng nhất là truyền riêng semantic_score và bm25_score.
            base_score = chunk.get("score", 0.0) 
            semantic_score = chunk.get("semantic_score", base_score)
            keyword_score = chunk.get("bm25_score", base_score)

            # Tính các điểm bổ sung
            source_score = self._score_source_quality(meta)
            completeness_score = self._score_completeness(chunk.get("text", ""))

            # TÍNH ĐIỂM TỔNG HỢP (WEIGHTED SCORE)
            final_score = (
                (semantic_score * WEIGHT_SEMANTIC) +
                (keyword_score * WEIGHT_KEYWORD) +
                (source_score * WEIGHT_SOURCE) +
                (completeness_score * WEIGHT_COMPLETENESS)
            )

            # Chỉ giữ lại bằng chứng đạt chất lượng
            if final_score >= MIN_VALID_SCORE:
                # Ghi đè điểm final để phục vụ xếp hạng
                chunk["validation_score"] = final_score
                chunk["source_quality_score"] = source_score
                validated_candidates.append(chunk)

        # Sắp xếp từ cao xuống thấp theo điểm Validation
        validated_candidates.sort(key=lambda x: x["validation_score"], reverse=True)

        # BƯỚC 2: LOẠI BỎ TRÙNG LẶP (DEDUPLICATION)
        # Vì đã xếp hạng từ cao xuống thấp, ta giữ lại đoạn điểm cao và loại đoạn điểm thấp bị trùng
        final_evidence = []
        for current_chunk in validated_candidates:
            is_duplicate = False
            for accepted_chunk in final_evidence:
                sim = self._calculate_jaccard(current_chunk["text"], accepted_chunk["text"])
                if sim > JACCARD_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_evidence.append(current_chunk)

        return final_evidence
