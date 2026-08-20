import numpy as np
import streamlit as st
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import google.generativeai as genai

import key_manager # Bắt buộc gọi trái tim chứa 8 Key ở đây

# ============================================================
# CẤU HÌNH MÔ HÌNH (Đã chuyển sang các bản Siêu Nhẹ)
# ============================================================

# Model Embedding siêu nhẹ (~80MB RAM)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Reranker siêu nhẹ (~90MB RAM) thay cho bản BAAI cũ (rất nặng CPU)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================
# 1. QUẢN LÝ MÔ HÌNH (EMBEDDING & RERANKER)
# ============================================================

@st.cache_resource
def load_embedding_model(model_name: str):
    """Tải mô hình nhúng (Vector) vào RAM và đóng băng bằng Streamlit cache."""
    return SentenceTransformer(model_name)

@st.cache_resource
def load_reranker_model(model_name: str):
    """Tải mô hình Cross-Encoder vào RAM (chỉ tải 1 lần)."""
    return CrossEncoder(model_name)

def get_embeddings(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Chuyển đổi danh sách văn bản thành ma trận vector ngữ nghĩa."""
    model = load_embedding_model(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)

# ============================================================
# 2. XÂY DỰNG CHỈ MỤC TÌM KIẾM (INDEX BUILDER)
# ============================================================

def _safe_get_text(chunk: Any) -> str:
    """Hàm hỗ trợ lấy text an toàn bất kể chunk là Dictionary hay Object."""
    return chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")

def build_bm25_index(chunks: List[Any]) -> BM25Okapi:
    """Dựng chỉ mục từ khóa truyền thống BM25."""
    tokenized_corpus = [_safe_get_text(c).lower().split() for c in chunks if _safe_get_text(c)]
    return BM25Okapi(tokenized_corpus) if tokenized_corpus else None

def build_embedding_index(chunks: List[Any], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Dựng ma trận vector cho toàn bộ văn bản."""
    texts = [_safe_get_text(c) for c in chunks if _safe_get_text(c)]
    if not texts:
        return np.array([])
    return get_embeddings(texts, model_name)

def update_embedding_index(new_chunks: List[Any], existing_matrix: np.ndarray, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Nối thêm vector của văn bản mới vào ma trận cũ."""
    new_texts = [_safe_get_text(c) for c in new_chunks if _safe_get_text(c)]
    if not new_texts:
        return existing_matrix
        
    new_matrix = get_embeddings(new_texts, model_name)
    if existing_matrix is not None and existing_matrix.size > 0:
        return np.vstack([existing_matrix, new_matrix])
    return new_matrix

# ============================================================
# 3. THUẬT TOÁN TRUY XUẤT 2 GIAI ĐOẠN (HYBRID + RERANKER)
# ============================================================

def retrieve_evidence(
    query: str, 
    chunks: List[Dict[str, Any]], 
    matrix: np.ndarray, 
    bm25: BM25Okapi, 
    top_k: int = 8, 
    hybrid_top_k: int = 12,  # Đã giảm từ 30 xuống 12 để cứu CPU
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    reranker_name: str = DEFAULT_RERANKER_MODEL
) -> List[Dict[str, Any]]:
    
    if not chunks or matrix is None or bm25 is None or len(chunks) == 0:
        return []

    # --- STAGE 1: HYBRID SEARCH ---
    query_vector = get_embeddings([query], model_name)[0]
    semantic_scores = matrix @ query_vector

    # An toàn hóa việc chuẩn hóa Semantic
    sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
    if sem_max > sem_min:
        semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min)
    else:
        semantic_scores = np.ones_like(semantic_scores)

    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    # An toàn hóa việc chuẩn hóa BM25
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max > bm25_min:
        bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_scores = np.ones_like(bm25_scores)

    # Tính điểm Hybrid
    final_scores = (0.65 * semantic_scores) + (0.35 * bm25_scores)
    
    stage1_k = min(hybrid_top_k, len(chunks))
    stage1_indices = np.argsort(final_scores)[::-1][:stage1_k]
    
    candidate_chunks = []
    for idx in stage1_indices:
        chunk_copy = dict(chunks[idx])
        chunk_copy["hybrid_score"] = float(final_scores[idx])
        candidate_chunks.append(chunk_copy)

    if len(candidate_chunks) <= top_k:
        for doc in candidate_chunks:
            doc["score"] = doc["hybrid_score"]
        return candidate_chunks[:top_k]

    # --- STAGE 2: CROSS-ENCODER RERANKING ---
    reranker = load_reranker_model(reranker_name)
    
    sentence_pairs = [[query, doc["text"]] for doc in candidate_chunks]
    rerank_scores = reranker.predict(sentence_pairs)
    
    for i, doc in enumerate(candidate_chunks):
        doc["score"] = float(rerank_scores[i])
        
    candidate_chunks.sort(key=lambda x: x["score"], reverse=True)

    return candidate_chunks[:top_k]

# ============================================================
# 4. GỌI GEMINI CHO TÍNH NĂNG TẠO TỪ KHÓA MESH (MỚI)
# ============================================================
def generate_mesh_keywords(query: str) -> str:
    """Sử dụng Gemini để tự động dịch và trích xuất từ khóa y khoa (MeSH)."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = f"""
Bạn là một chuyên gia thư viện y khoa.
Người dùng đang muốn tìm kiếm tài liệu nghiên cứu về chủ đề sau: "{query}"
Hãy chuyển đổi chủ đề này thành 1 câu truy vấn tiếng Anh duy nhất, sử dụng các thuật ngữ y khoa MeSH chuẩn nhất.
KHÔNG giải thích, CHỈ trả về đúng 1 dòng chứa câu truy vấn tiếng Anh.
"""
        response = None
        for attempt in range(2):
            try:
                # Gọi Key từ trái tim của hệ thống
                api_key = key_manager.get_next_key()
                if not api_key:
                    return query # Fallback
                genai.configure(api_key=api_key)
                
                response = model.generate_content(prompt)
                break
            except Exception as inner_e:
                if ("429" in str(inner_e) or "Quota" in str(inner_e)) and attempt == 0:
                    continue 
                else:
                    return query
                    
        return response.text.strip() if response else query
    except Exception:
        return query
