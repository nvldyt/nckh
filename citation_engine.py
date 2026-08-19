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
        Quét bản nháp, đánh số Vancouver linh hoạt (hỗ trợ cả gộp trích dẫn [REF-1, REF-2])
        và trả về văn bản hoàn chỉnh cùng danh mục TLTK.
        """
        # 1. Quét tìm TẤT CẢ các mã REF- xuất hiện trong bài (dù đứng lẻ hay đứng chung)
        # Sử dụng regex tìm mã gốc thay vì tìm cả cụm ngoặc vuông
        all_ref_matches = re.findall(r"(REF-[a-zA-Z0-9_-]+)", draft_text)
        
        vancouver_mapping = {}
        current_index = 1
        invalid_citations = []
        
        # 2. Xây dựng Mapping (Đánh số ưu tiên theo thứ tự xuất hiện)
        for ref_id in all_ref_matches:
            if ref_id not in vancouver_mapping:
                # Kiểm tra Validation
                if ref_id in self.source_registry:
                    vancouver_mapping[ref_id] = current_index
                    current_index += 1
                else:
                    vancouver_mapping[ref_id] = "INVALID"
                    invalid_citations.append(ref_id)
                    logging.warning(f"Cảnh báo: Phát hiện trích dẫn ảo -> {ref_id}")

        # 3. Hàm xử lý linh hoạt mọi loại ngoặc chứa REF
        def replace_tag_group(match):
            inner_text = match.group(1) # Lấy phần chữ bên trong ngoặc vuông
            # Bóc tách tất cả các mã REF- có bên trong cụm ngoặc này
            refs_in_bracket = re.findall(r"(REF-[a-zA-Z0-9_-]+)", inner_text)
            
            if not refs_in_bracket:
                return match.group(0) # Trả lại nguyên vẹn nếu bắt nhầm ngoặc không liên quan
                
            replaced_nums = []
            for ref in refs_in_bracket:
                val = vancouver_mapping.get(ref)
                if val == "INVALID":
                    replaced_nums.append("⚠️ LỖI ẢO")
                elif val is not None:
                    replaced_nums.append(str(val))
            
            # Ghép lại chuẩn Vancouver: ví dụ [1, 2]
            return f"[{', '.join(replaced_nums)}]"

        # Thay thế mọi ngoặc vuông CÓ CHỨA mã REF-
        # Pattern này bắt mọi thứ dạng [...REF-...]
        final_text = re.sub(r"\[([^\]]*REF-[^\]]*)\]", replace_tag_group, draft_text)
        
        # 4. Xuất danh sách tài liệu tham khảo đã sắp xếp
        final_citations = []
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
