# retrieval_engine.py
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Khai báo model mặc định ở đây để quản lý tập trung
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================
# 1. QUẢN LÝ MÔ HÌNH NHÚNG (EMBEDDING MODEL)
# ============================================================

@lru_cache(maxsize=1)
def load_embedding_model(model_name: str):
    """
    Tải mô hình nhúng và lưu vào RAM (cache) để không phải tải lại mỗi lần gọi.
    Chỉ giữ tối đa 1 model trong bộ nhớ để tránh tràn RAM.
    """
    return SentenceTransformer(model_name)

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
    """Dựng ma trận vector cho toàn bộ văn bản (Dùng khi nạp mới hoàn toàn)."""
    texts = [c["text"] for c in chunks]
    return get_embeddings(texts, model_name)

def update_embedding_index(new_chunks: List[Dict[str, Any]], existing_matrix: np.ndarray, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Nối thêm vector của văn bản mới vào ma trận cũ để tiết kiệm thời gian tính toán."""
    new_texts = [c["text"] for c in new_chunks]
    new_matrix = get_embeddings(new_texts, model_name)
    if existing_matrix is not None and existing_matrix.size > 0:
        return np.vstack([existing_matrix, new_matrix])
    return new_matrix

# ============================================================
# 3. THUẬT TOÁN TRUY XUẤT (HYBRID RETRIEVAL)
# ============================================================

def retrieve_evidence(
    query: str, 
    chunks: List[Dict[str, Any]], 
    matrix: np.ndarray, 
    bm25: BM25Okapi, 
    k: int = 8, 
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> List[Dict[str, Any]]:
    """
    Thuật toán Hybrid Search: Trộn điểm Vector (65%) và BM25 (35%).
    (Sẵn sàng tích hợp Reranker ở bước này trong tương lai).
    """
    if not chunks or matrix is None or bm25 is None:
        return []

    # 1. Semantic Search (Đo khoảng cách Vector)
    query_vector = get_embeddings([query], model_name)[0]
    semantic_scores = matrix @ query_vector

    sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
    if sem_max > sem_min:
        semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min)
    else:
        semantic_scores = np.zeros_like(semantic_scores)

    # 2. Keyword Search (Đo độ khớp từ khóa bằng BM25)
    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max > bm25_min:
        bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_scores = np.zeros_like(bm25_scores)

    # 3. Hybrid Scoring
    final_scores = (0.65 * semantic_scores) + (0.35 * bm25_scores)
    
    # 4. Lấy Top-K ứng viên xuất sắc nhất
    # (Tương lai: Có thể lấy Top-30, rồi dùng Reranker lọc lại Top-8 tại vị trí này)
    k = min(k, len(chunks))
    indices = np.argsort(final_scores)[::-1][:k]

    results = []
    for idx in indices:
        item = dict(chunks[idx])  # Sao chép để không ảnh hưởng dữ liệu gốc
        item["score"] = float(final_scores[idx])
        results.append(item)

    return results

# Lưu ý: Các hàm liên lạc API ngoài (PubMed, VN Journals) đã được giữ 
# ở file `evidence_engine.py` vì chúng thuộc nhóm "Nạp dữ liệu", 
# còn file này chuyên trị về "Tính toán & Truy xuất nội bộ".
