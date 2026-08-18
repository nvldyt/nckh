import re
import logging

class CitationEngine:
    def __init__(self):
        # Lưu trữ mapping từ mã nguồn sang metadata gốc
        # Dữ liệu dạng: { "REF-001": {"title": "...", "authors": "...", "doi": "..."} }
        self.source_registry = {}

    def register_evidence(self, source_id: str, metadata: dict) -> str:
        """
        Nạp tài liệu vào registry và trả về mã định danh để đưa vào prompt cho AI.
        """
        # Đảm bảo mã luôn có tiền tố REF- để Regex dễ dàng nhận diện
        ref_id = f"REF-{source_id}" if not str(source_id).startswith("REF-") else str(source_id)
        self.source_registry[ref_id] = metadata
        return f"[{ref_id}]"

    def process_vancouver_citations(self, draft_text: str) -> tuple[str, list, list]:
        """
        Quét bản nháp, đánh số Vancouver và trả về văn bản hoàn chỉnh cùng danh sách tham khảo.
        """
        # Regex tìm tất cả các chuỗi có dạng [REF-...]
        pattern = r"\[(REF-[a-zA-Z0-9_-]+)\]"
        
        # Tìm các tag theo ĐÚNG thứ tự xuất hiện trong văn bản
        matches = re.findall(pattern, draft_text)
        
        vancouver_mapping = {}
        current_index = 1
        invalid_citations = []
        
        # 1. Đánh số thứ tự dựa trên lần xuất hiện đầu tiên
        for ref_id in matches:
            if ref_id not in vancouver_mapping:
                # Citation Validator: Kiểm tra xem mã này có thực sự tồn tại trong registry không
                if ref_id in self.source_registry:
                    vancouver_mapping[ref_id] = current_index
                    current_index += 1
                else:
                    # Ghi nhận lỗi nếu AI "ảo giác" (Hallucination) ra một mã không tồn tại
                    vancouver_mapping[ref_id] = "INVALID"
                    invalid_citations.append(ref_id)
                    logging.warning(f"Cảnh báo: Phát hiện trích dẫn ảo không có thật -> {ref_id}")

        # 2. Hàm thay thế mã tag bằng số Vancouver
        def replace_tag(match):
            ref_id = match.group(1)
            val = vancouver_mapping.get(ref_id)
            if val == "INVALID":
                return "[⚠️ LỖI TRÍCH DẪN ÁO]"
            elif val is not None:
                return f"[{val}]"
            return match.group(0) # Trả về nguyên trạng nếu có lỗi rủi ro
            
        # Biến đổi văn bản nháp thành văn bản chính thức
        final_text = re.sub(pattern, replace_tag, draft_text)
        
        # 3. Xuất danh sách tài liệu tham khảo theo đúng thứ tự 1, 2, 3...
        final_citations = []
        # Lọc bỏ các tag lỗi trước khi sắp xếp
        valid_mappings = {k: v for k, v in vancouver_mapping.items() if v != "INVALID"}
        sorted_mappings = sorted(valid_mappings.items(), key=lambda item: item[1])
        
        for ref_id, index in sorted_mappings:
            metadata = self.source_registry.get(ref_id, {})
            final_citations.append({
                "vancouver_index": index,
                "ref_id": ref_id,
                "metadata": metadata
            })
            
        return final_text, final_citations, invalid_citations

    def clear_registry(self):
        """Dọn dẹp registry khi bắt đầu một phiên làm việc mới."""
        self.source_registry.clear()
