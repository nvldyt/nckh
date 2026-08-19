# retrieval_engine.py
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Khai báo model mặc định ở đây để quản lý tập trung
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Khai báo Reranker đa ngôn ngữ (Nhỏ, nhẹ nhưng cực kỳ thông minh)
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ============================================================
# 1. QUẢN LÝ MÔ HÌNH (EMBEDDING & RERANKER)
# ============================================================

@lru_cache(maxsize=1)
def load_embedding_model(model_name: str):
    """Tải mô hình nhúng (Vector) vào RAM."""
    return SentenceTransformer(model_name)

@lru_cache(maxsize=1)
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

def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """Dựng chỉ mục từ khóa truyền thống BM25."""
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized_corpus)

def build_embedding_index(chunks: List[Dict[str, Any]], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Dựng ma trận vector cho toàn bộ văn bản."""
    texts = [c["text"] for c in chunks]
    return get_embeddings(texts, model_name)

def update_embedding_index(new_chunks: List[Dict[str, Any]], existing_matrix: np.ndarray, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Nối thêm vector của văn bản mới vào ma trận cũ."""
    new_texts = [c["text"] for c in new_chunks]
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
    hybrid_top_k: int = 30, # Lấy 30 ứng viên để Rerank
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    reranker_name: str = DEFAULT_RERANKER_MODEL
) -> List[Dict[str, Any]]:
    """
    Tìm kiếm bằng chứng thông minh (Agentic Retrieval):
    - Giai đoạn 1: Lọc thô bằng Hybrid (Vector + BM25) lấy Top 30.
    - Giai đoạn 2: Cross-Encoder Reranker soi xét lại Top 30 để chắt lọc Top 8 chuẩn nhất.
    """
    if not chunks or matrix is None or bm25 is None:
        return []

    # --- STAGE 1: HYBRID SEARCH (Lọc thô Top 30) ---
    query_vector = get_embeddings([query], model_name)[0]
    semantic_scores = matrix @ query_vector

    sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
    if sem_max > sem_min:
        semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min)
    else:
        semantic_scores = np.zeros_like(semantic_scores)

    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max > bm25_min:
        bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_scores = np.zeros_like(bm25_scores)

    final_scores = (0.65 * semantic_scores) + (0.35 * bm25_scores)
    
    stage1_k = min(hybrid_top_k, len(chunks))
    stage1_indices = np.argsort(final_scores)[::-1][:stage1_k]
    
    candidate_chunks = [dict(chunks[idx]) for idx in stage1_indices]

    # NẾU DỮ LIỆU CÓ ÍT HƠN SỐ LƯỢNG YÊU CẦU, TRẢ VỀ LUÔN KHÔNG CẦN RERANK
    if len(candidate_chunks) <= top_k:
        return candidate_chunks[:top_k]

    # --- STAGE 2: CROSS-ENCODER RERANKING (Tinh chỉnh Top K) ---
    reranker = load_reranker_model(reranker_name)
    
    # Tạo cặp câu (Query, Document_Text) để chấm điểm chéo
    sentence_pairs = [[query, doc["text"]] for doc in candidate_chunks]
    rerank_scores = reranker.predict(sentence_pairs)
    
    # Cập nhật điểm chuẩn xác từ Reranker và sắp xếp lại
    for i, doc in enumerate(candidate_chunks):
        doc["score"] = float(rerank_scores[i]) 
        
    candidate_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Chỉ trả về Top K tinh túy nhất cho AI
    return candidate_chunks[:top_k]
