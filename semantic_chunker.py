# semantic_chunker.py
import re
from typing import List, Tuple

class SemanticChunker:
    # Điều chỉnh max_chunk_size nhỏ lại một chút (1000-1200) là tối ưu nhất cho Embedding
    def __init__(self, max_chunk_size: int = 1200, min_chunk_size: int = 100, chunk_overlap: int = 250):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap # Số ký tự gối đầu giữa các đoạn
        
        # Regex nhận diện tiêu đề y khoa điển hình
        self.heading_pattern = re.compile(
            r'^(Mục\s+\d+|[IVXLCDM]+\.|[0-9]+(\.[0-9]+)*\.*)\s+[A-ZÀ-Ỹa-zà-ỹ].*', 
            re.MULTILINE
        )

    def clean_text(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_by_semantics(self, text: str) -> List[Tuple[str, int, int]]:
        text = self.clean_text(text)
        if not text:
            return []

        chunks = []
        
        # Bước 1: Trích xuất các Tiêu đề và nội dung tương ứng
        sections = []
        last_pos = 0
        current_heading = "Nội dung chung" # Tiêu đề mặc định nếu đoạn đầu không có heading
        
        matches = list(self.heading_pattern.finditer(text))
        for i, match in enumerate(matches):
            if i == 0 and match.start() > 0:
                # Phần văn bản mở đầu trước khi có tiêu đề đầu tiên
                sections.append((current_heading, text[0:match.start()], 0, match.start()))
            
            current_heading = match.group().strip()
            start_pos = match.start()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            sections.append((current_heading, text[start_pos:end_pos], start_pos, end_pos))

        # Bước 2: Xử lý từng section với Sliding Window & Overlap
        for heading, section_text, sec_start, sec_end in sections:
            if len(section_text) <= self.max_chunk_size:
                if len(section_text) >= self.min_chunk_size:
                    chunks.append((section_text.strip(), sec_start, sec_end))
            else:
                # Đoạn quá dài -> Cắt theo paragraph và áp dụng gối đầu (Overlap)
                paragraphs = section_text.split('\n\n')
                current_paragraphs = []
                current_len = 0
                
                for p in paragraphs:
                    if current_len + len(p) <= self.max_chunk_size:
                        current_paragraphs.append(p)
                        current_len += len(p) + 2
                    else:
                        # Lưu chunk hiện tại
                        chunk_text = "\n\n".join(current_paragraphs).strip()
                        if len(chunk_text) >= self.min_chunk_size:
                            # TIÊM NGỮ CẢNH: Đảm bảo chunk luôn chứa tiêu đề nếu nó bị cắt mất
                            if heading not in chunk_text:
                                chunk_text = f"[{heading}]\n{chunk_text}"
                            chunks.append((chunk_text, sec_start, sec_start + len(chunk_text)))
                        
                        # Tạo OVERLAP: Giữ lại một vài đoạn văn cuối của chunk trước
                        overlap_len = 0
                        overlap_paragraphs = []
                        for cp in reversed(current_paragraphs):
                            if overlap_len + len(cp) > self.chunk_overlap and len(overlap_paragraphs) > 0:
                                break
                            overlap_paragraphs.insert(0, cp)
                            overlap_len += len(cp) + 2
                            
                        # Bắt đầu chunk mới với phần gối đầu + đoạn văn hiện tại
                        current_paragraphs = overlap_paragraphs + [p]
                        current_len = sum(len(x) + 2 for x in current_paragraphs)
                        # Cập nhật sec_start giả định cho chunk tiếp theo
                        sec_start += len(chunk_text) - overlap_len 
                
                # Xử lý phần còn dư cuối cùng
                if current_paragraphs:
                    final_chunk = "\n\n".join(current_paragraphs).strip()
                    if len(final_chunk) >= self.min_chunk_size:
                        if heading not in final_chunk:
                            final_chunk = f"[{heading}]\n{final_chunk}"
                        chunks.append((final_chunk, sec_start, sec_start + len(final_chunk)))

        return chunks
