import numpy as np
import streamlit as st
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Import module quản lý Key của bạn
import key_manager 

# ============================================================
# CẤU HÌNH MÔ HÌNH (Đã chuyển sang các bản Siêu Nhẹ)
# ============================================================

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================
# HÀM LẤY API KEY (BẢN OFFLINE/FILE-BASED)
# ============================================================

def get_serpapi_key() -> Optional[str]:
    """Lấy API Key cho việc tìm kiếm bài báo (SerpAPI)."""
    # 1. Ưu tiên đọc từ file serpapi.txt
    if os.path.exists("serpapi.txt"):
        try:
            with open("serpapi.txt", "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key: return key
        except Exception: pass
    
    # 2. Key mặc định cho SerpAPI (để không bị lỗi)
    return "f99c73f0a83c6e0ec159f8583534aa2d9deabdd339c44511323b83c15c4c6704"

def get_default_gemini_key() -> str:
    """Hàm lấy Key dự phòng cho Gemini nếu key_manager bị lỗi."""
    return "AQ.Ab8RN6JhBJ5w9bnl4pcVuf_NBh8gb2pwRq756ybmvXnar9Q18A"

# ============================================================
# 1. QUẢN LÝ MÔ HÌNH (EMBEDDING & RERANKER)
# ============================================================

@st.cache_resource
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_reranker_model(model_name: str):
    return CrossEncoder(model_name)

def get_embeddings(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    model = load_embedding_model(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)

# ============================================================
# 2. XÂY DỰNG CHỈ MỤC TÌM KIẾM (KHÔNG THAY ĐỔI)
# ============================================================

def _safe_get_text(chunk: Any) -> str:
    return chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")

def build_bm25_index(chunks: List[Any]) -> BM25Okapi:
    tokenized_corpus = [_safe_get_text(c).lower().split() for c in chunks if _safe_get_text(c)]
    return BM25Okapi(tokenized_corpus) if tokenized_corpus else None

def build_embedding_index(chunks: List[Any], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    texts = [_safe_get_text(c) for c in chunks if _safe_get_text(c)]
    if not texts:
        return np.array([])
    return get_embeddings(texts, model_name)

def update_embedding_index(new_chunks: List[Any], existing_matrix: np.ndarray, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    new_texts = [_safe_get_text(c) for c in new_chunks if _safe_get_text(c)]
    if not new_texts:
        return existing_matrix
        
    new_matrix = get_embeddings(new_texts, model_name)
    if existing_matrix is not None and existing_matrix.size > 0:
        return np.vstack([existing_matrix, new_matrix])
    return new_matrix

# ============================================================
# 3. THUẬT TOÁN TRUY XUẤT 2 GIAI ĐOẠN (KHÔNG THAY ĐỔI)
# ============================================================

def retrieve_evidence(
    query: str, 
    chunks: List[Dict[str, Any]], 
    matrix: np.ndarray, 
    bm25: BM25Okapi, 
    top_k: int = 8, 
    hybrid_top_k: int = 12,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    reranker_name: str = DEFAULT_RERANKER_MODEL
) -> List[Dict[str, Any]]:
    
    if not chunks or matrix is None or bm25 is None or len(chunks) == 0:
        return []

    query_vector = get_embeddings([query], model_name)[0]
    semantic_scores = matrix @ query_vector

    sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
    if sem_max > sem_min:
        semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min)
    else:
        semantic_scores = np.ones_like(semantic_scores)

    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max > bm25_min:
        bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_scores = np.ones_like(bm25_scores)

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

    reranker = load_reranker_model(reranker_name)
    sentence_pairs = [[query, doc["text"]] for doc in candidate_chunks]
    rerank_scores = reranker.predict(sentence_pairs)
    
    for i, doc in enumerate(candidate_chunks):
        doc["score"] = float(rerank_scores[i])
        
    candidate_chunks.sort(key=lambda x: x["score"], reverse=True)

    return candidate_chunks[:top_k]
