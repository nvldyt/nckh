import io
import re
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import requests
import xml.etree.ElementTree as ET
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ============================================================
# 1. CẤU TRÚC DỮ LIỆU BẰNG CHỨNG
# ============================================================

@dataclass
class SourceDocument:
    source_id: str
    file_name: str
    file_hash: str
    origin: str = "PDF"          # PDF | PubMed | Tạp chí VN
    title: str = ""
    authors: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""

@dataclass
class EvidenceChunk:
    chunk_id: str
    source_id: str
    file_name: str
    page: int
    text: str
    char_start: int
    char_end: int
    section: str = ""
    table_hint: str = ""

# ============================================================
# 2. TIỆN ÍCH CHUNG: HASH & TẠO ID
# ============================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def make_source_id(file_name: str, file_hash: str) -> str:
    raw = f"{file_name}|{file_hash}"
    return "SRC-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10].upper()

def make_chunk_id(source_id: str, page: int, index: int) -> str:
    return f"{source_id}-P{page:03d}-C{index:03d}"

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ============================================================
# 3. THUẬT TOÁN CHIA NHỎ TÀI LIỆU (CHUNKING)
# ============================================================

def split_text_into_chunks(
    text: str, chunk_size: int = 1800, overlap: int = 300
) -> List[Tuple[str, int, int]]:
    """
    Cắt text thành các đoạn nhỏ (chunk) dựa trên số ký tự.
    Thuật toán sẽ cố gắng lùi lại tìm dấu xuống dòng kép (\n\n) 
    hoặc dấu chấm câu (. ) để cắt, tránh làm đứt gãy câu chữ giữa chừng.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            # Ưu tiên cắt ở ngắt đoạn
            candidate = text.rfind("\n\n", start, end)
            if candidate > start + int(chunk_size * 0.55):
                end = candidate
            else:
                # Nếu không có ngắt đoạn, cắt ở dấu chấm kết câu
                candidate = text.rfind(". ", start, end)
                if candidate > start + int(chunk_size * 0.55):
                    end = candidate + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start, end))

        if end >= n:
            break
        
        # Lùi lại tạo độ phủ (overlap) để không mất ngữ cảnh giữa 2 chunk
        start = max(end - overlap, start + 1)

    return chunks

# ============================================================
# 4. EMBEDDING (VECTORIZATION) & INDEXING
# ============================================================

MEDICAL_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]+(?:[-/.][A-Za-zÀ-ỹ0-9]+)*")

def medical_tokenize(text: str) -> List[str]:
    return MEDICAL_TOKEN_RE.findall(text.lower().strip())

@st.cache_resource
def load_embedding_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return SentenceTransformer(model_name)

def get_embeddings(texts: List[str]) -> np.ndarray:
    model = load_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)

def rebuild_index(new_chunks: List[EvidenceChunk] = None):
    """
    Tính toán lại hoặc cập nhật Vector (Embeddings) và BM25 Index 
    khi có tài liệu mới được nạp vào.
    """
    all_chunks = st.session_state.get("chunks", [])
    if not all_chunks:
        st.session_state["embeddings"] = None
        st.session_state["bm25"] = None
        return

    old_embeddings = st.session_state.get("embeddings")
    
    # Tối ưu: Nếu đã có vector cũ, chỉ tính vector cho các chunk mới rồi nối vào (np.vstack)
    if new_chunks and old_embeddings is not None and len(old_embeddings) == len(all_chunks) - len(new_chunks):
        new_texts = [c["text"] for c in new_chunks]
        new_matrix = get_embeddings(new_texts)
        st.session_state["embeddings"] = np.vstack([old_embeddings, new_matrix])
    else:
        # Nếu không, tính lại từ đầu
        texts = [c["text"] for c in all_chunks]
        st.session_state["embeddings"] = get_embeddings(texts)

    # Cập nhật thuật toán tìm kiếm từ khóa BM25
    st.session_state["bm25"] = BM25Okapi([medical_tokenize(c["text"]) for c in all_chunks])

def add_source_and_chunks(source: SourceDocument, chunks: List[EvidenceChunk]) -> bool:
    if source.source_id in st.session_state.get("documents", {}):
        return False
        
    st.session_state["documents"][source.source_id] = asdict(source)
    st.session_state["chunks"].extend([asdict(c) for c in chunks])
    return True

# ============================================================
# 5. CÁC HÀM NẠP TÀI LIỆU (PDF & API)
# ============================================================

def extract_pdf(uploaded_file) -> Tuple[SourceDocument, List[EvidenceChunk]]:
    """Đọc file PDF, bóc tách text và chia thành các vector chunks."""
    data = uploaded_file.getvalue()
    file_hash = sha256_bytes(data)
    source_id = make_source_id(uploaded_file.name, file_hash)

    reader = PdfReader(io.BytesIO(data))
    source = SourceDocument(
        source_id=source_id,
        file_name=uploaded_file.name,
        file_hash=file_hash,
        origin="PDF",
    )

    chunks: List[EvidenceChunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""

        text = clean_text(raw)
        if not text:
            continue

        for idx, (piece, start, end) in enumerate(split_text_into_chunks(text), start=1):
            chunks.append(
                EvidenceChunk(
                    chunk_id=make_chunk_id(source_id, page_no, idx),
                    source_id=source_id,
                    file_name=uploaded_file.name,
                    page=page_no,
                    text=piece,
                    char_start=start,
                    char_end=end,
                )
            )

    return source, chunks
