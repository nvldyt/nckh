# semantic_chunker.py
import re
from typing import List, Tuple

class SemanticChunker:
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Regex nhận diện tiêu đề y khoa điển hình (Ví dụ: "1. Tổng quan", "II. CHỈ ĐỊNH", "2.1. Liều dùng")
        self.heading_pattern = re.compile(
            r'^(Mục\s+\d+|[IVXLCDM]+\.|[0-9]+(\.[0-9]+)*\.*)\s+[A-ZÀ-Ỹa-zà-ỹ]', 
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
        
        # Bước 1: Chia văn bản theo các Tiêu đề (Headings)
        sections = []
        last_pos = 0
        
        for match in self.heading_pattern.finditer(text):
            if match.start() > last_pos:
                sections.append((text[last_pos:match.start()], last_pos, match.start()))
            last_pos = match.start()
        
        if last_pos < len(text):
            sections.append((text[last_pos:], last_pos, len(text)))

        # Bước 2: Xử lý từng section. Nếu section quá dài, chia tiếp theo đoạn văn (\n\n)
        for section_text, sec_start, sec_end in sections:
            if len(section_text) <= self.max_chunk_size:
                if len(section_text) >= self.min_chunk_size:
                    chunks.append((section_text.strip(), sec_start, sec_end))
            else:
                # Cắt theo đoạn văn
                paragraphs = section_text.split('\n\n')
                current_chunk = ""
                current_start = sec_start
                
                for p in paragraphs:
                    if len(current_chunk) + len(p) <= self.max_chunk_size:
                        current_chunk += p + "\n\n"
                    else:
                        if len(current_chunk) >= self.min_chunk_size:
                            chunks.append((current_chunk.strip(), current_start, current_start + len(current_chunk)))
                        current_start += len(current_chunk)
                        current_chunk = p + "\n\n"
                        
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append((current_chunk.strip(), current_start, current_start + len(current_chunk)))

        return chunks
