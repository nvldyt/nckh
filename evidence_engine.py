# File: evidence_engine.py (Bản OFFLINE - Tối ưu RAM & Đồng bộ Key)

import io
import re
import os
import hashlib
import concurrent.futures
import gc  # Thêm thư viện dọn rác RAM
from serpapi import GoogleSearch
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from semantic_chunker import SemanticChunker

import requests
import xml.etree.ElementTree as ET
import streamlit as st
import fitz

import key_manager # Bắt buộc gọi trái tim chứa 8 Key ở đây
import google.generativeai as genai # Gọi thư viện Gemini

# ============================================================
# 1. CẤU TRÚC DỮ LIỆU BẰNG CHỨNG
# ============================================================
@dataclass
class SourceDocument:
    source_id: str
    file_name: str
    file_hash: str
    origin: str = "PDF"
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
def split_text_into_chunks(text: str, chunk_size: int = 1800, overlap: int = 300) -> List[Tuple[str, int, int]]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            candidate = text.rfind("\n\n", start, end)
            if candidate > start + int(chunk_size * 0.55):
                end = candidate
            else:
                candidate = text.rfind(". ", start, end)
                if candidate > start + int(chunk_size * 0.55):
                    end = candidate + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks

# ============================================================
# 4. QUẢN LÝ STATE TÀI LIỆU
# ============================================================
def add_source_and_chunks(source: SourceDocument, chunks: List[EvidenceChunk]) -> bool:
    if source.source_id in st.session_state.get("documents", {}):
        return False
    st.session_state["documents"][source.source_id] = asdict(source)
    st.session_state["chunks"].extend([asdict(c) for c in chunks])
    return True

# ============================================================
# 5. XỬ LÝ PDF (Bằng PyMuPDF - Xử lý thông minh 2 cột và bảng biểu)
# ============================================================
def extract_pdf(uploaded_file) -> Tuple[SourceDocument, List[EvidenceChunk]]:
    data = uploaded_file.getvalue()
    file_hash = sha256_bytes(data)
    source_id = make_source_id(uploaded_file.name, file_hash)

    source = SourceDocument(
        source_id=source_id,
        file_name=uploaded_file.name,
        file_hash=file_hash,
        origin="PDF",
    )

    chunks: List[EvidenceChunk] = []
    
    try:
        # Bơm Key từ module Offline cho Semantic Chunker (Dùng cho Embedding)
        api_key = key_manager.get_next_key()
        if api_key and api_key != "CHUA_CO_KEY":
            genai.configure(api_key=api_key)
            
        chunker = SemanticChunker(max_chunk_size=1200, min_chunk_size=100, chunk_overlap=250)
        
        # Dùng 'with' để tự động đóng file PDF, chống kẹt RAM
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page_no in range(len(doc)):
                page = doc[page_no]
                raw = page.get_text("text", sort=True) or ""
                
                if not raw.strip():
                    continue

                semantic_pieces = chunker.split_by_semantics(raw)
                
                for idx, (piece, start, end) in enumerate(semantic_pieces, start=1):
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=make_chunk_id(source_id, page_no + 1, idx),
                            source_id=source_id,
                            file_name=uploaded_file.name,
                            page=page_no + 1,
                            text=piece,
                            char_start=start,
                            char_end=end,
                        )
                    )
    except Exception as e:
        st.error(f"Lỗi khi đọc file {uploaded_file.name}: {str(e)}")
    finally:
        # Ép dọn rác, thu hồi RAM ngay sau khi đọc PDF xong
        del data
        gc.collect()

    return source, chunks
    
# ============================================================
# 6. TRA CỨU API (PUBMED & VN JOURNALS)
# ============================================================
def get_serpapi_key() -> Optional[str]:
    """
    Lấy API Key cho việc tìm kiếm bài báo. 
    Ưu tiên lấy từ key_manager, nếu không có thì dùng key mặc định.
    Gỡ bỏ hoàn toàn st.secrets để không bị lỗi màn hình.
    """
    # 1. Thử lấy từ module key_manager (cách làm chuẩn)
    try:
        # Gọi hàm lấy key trong file key_manager.py
        key = key_manager.get_serpapi_key()
        if key and key != "CHUA_CO_KEY":
            return key
    except Exception:
        pass # Nếu file key_manager bị lỗi thì bỏ qua, xuống bước 2

    # 2. Key cứng dự phòng (Để app luôn chạy được kể cả khi không tìm thấy file text)
    return "f99c73f0a83c6e0ec159f8583534aa2d9deabdd339c44511323b83c15c4c6704"

def search_pubmed(query_en: str, max_res: int = 5) -> List[Dict[str, Any]]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query_en, "retmode": "json", "retmax": max_res}
    try:
        res = requests.get(search_url, params=params, timeout=20).json()
        id_list = res.get("esearchresult", {}).get("idlist", [])
    except Exception as exc:
        st.error(f"Lỗi tìm PubMed: {exc}")
        return []

    if not id_list:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
    try:
        res = requests.get(fetch_url, params=fetch_params, timeout=20)
    except Exception as exc:
        st.error(f"Lỗi tải chi tiết PubMed: {exc}")
        return []

    articles = []
    if res.status_code == 200:
        root = ET.fromstring(res.content)
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            title_node = article.find(".//ArticleTitle")
            pmid = pmid_node.text if pmid_node is not None else ""
            title = title_node.text if title_node is not None else "Không có tiêu đề"

            abstracts = article.findall(".//AbstractText")
            abs_text = " ".join([e.text for e in abstracts if e.text])

            author_node = article.find(".//Author/LastName")
            author = author_node.text if author_node is not None else "Không rõ"
            year_node = article.find(".//PubDate/Year")
            year = year_node.text if year_node is not None else ""

            journal_node = article.find(".//Journal/Title")
            journal = journal_node.text if journal_node is not None else ""

            doi = ""
            for eid in article.findall(".//ArticleId"):
                if eid.get("IdType") == "doi":
                    doi = eid.text or ""

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abs_text if abs_text else "Không có bản tóm tắt.",
                "authors": f"{author} và cộng sự" if author != "Không rõ" else author,
                "year": year,
                "journal": journal,
                "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    return articles

def generate_pubmed_queries(main_query: str) -> List[str]:
    """Tự động sinh ra 4 biến thể truy vấn song song để mở rộng phạm vi tìm kiếm PubMed."""
    clean_q = main_query.replace('"', '').strip()
    return [
        f"{clean_q}",
        f"{clean_q} AND (drug utilization OR treatment pattern OR management)",
        f"{clean_q} AND (clinical outcome OR blood pressure control OR effectiveness)",
        f"{clean_q} AND (outpatient OR inpatient OR prevalence OR risk factors)"
    ]

def search_pubmed_multi(main_query: str, max_res_per_query: int = 3) -> List[Dict[str, Any]]:
    """Thực hiện tìm kiếm đa biến thể SONG SONG, gom kết quả và lọc trùng bằng PMID tự động."""
    queries = generate_pubmed_queries(main_query)
    seen_pmids = set()
    all_articles = []
    
    # Bắn 4 request cùng lúc thay vì đợi tuần tự
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_query = {executor.submit(search_pubmed, q, max_res_per_query): q for q in queries}
        
        for future in concurrent.futures.as_completed(future_to_query):
            try:
                articles = future.result()
                for art in articles:
                    pmid = art.get("pmid")
                    if pmid and pmid not in seen_pmids:
                        seen_pmids.add(pmid)
                        all_articles.append(art)
                    elif not pmid:
                        all_articles.append(art)
            except Exception as exc:
                st.error(f"Lỗi truy vấn luồng PubMed: {exc}")
                
    return all_articles[:12]

def _sanitize_query(raw_query: str) -> str:
    if not raw_query:
        return raw_query
    cleaned = raw_query.replace("**", " ").replace("*", " ")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def _classify_vn_source(link: str) -> str:
    lower_link = (link or "").lower()
    if "vjol.info" in lower_link:
        return "Vietnam Journals Online (VJOL)"
    if "tapchiyhocvietnam.vn" in lower_link:
        return "Tạp chí Y học Việt Nam"
    if "jmpm.vn" in lower_link:
        return "Tạp chí Y Dược học Quân sự"
    if "huejmp.vn" in lower_link:
        return "Tạp chí Y Dược Huế"
    if "benhvien108" in lower_link:
        return "Tạp chí Y Dược lâm sàng 108"
    if "hup.edu.vn" in lower_link:
        return "Đại học Dược Hà Nội"
    return "Nghiên cứu Y học Việt Nam"

def _run_google_search(params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
    except Exception as e:
        return [], f"Lỗi kết nối SerpAPI: {str(e)}"

    if "error" in results:
        return [], f"SerpAPI báo lỗi: {results['error']}"

    return results.get("organic_results", []), None

def search_vn_journals(query: str, max_results: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    api_key = get_serpapi_key()
    if not api_key:
        return [], "⚠️ Chưa cấu hình SerpAPI Key (Tra cứu Tạp chí VN đang bị tắt)."

    query = _sanitize_query(query)
    if not query:
        return [], "Từ khóa rỗng sau khi làm sạch, vui lòng nhập lại."

    seen_links: set = set()
    collected_results: List[Dict[str, Any]] = []

    search_query = f"nghiên cứu {query} (tạp chí OR y học OR dược OR vjol OR pdf)"
    params_main = {
        "engine": "google",
        "q": search_query,
        "api_key": api_key,
        "hl": "vi",
        "gl": "vn",
        "num": max_results,
    }

    organic_results, error_msg = _run_google_search(params_main)
    if error_msg:
        return [], error_msg

    for item in organic_results:
        link = item.get("link", "")
        if not link or link in seen_links:
            continue
        seen_links.add(link)

        collected_results.append({
            "title": item.get("title", "Không có tiêu đề"),
            "link": link,
            "snippet": item.get("snippet", "Không có tóm tắt."),
            "source": _classify_vn_source(link),
            "origin": "Tạp chí VN",
        })

    if not collected_results:
        params_fallback = {
            "engine": "google",
            "q": f"nghiên cứu y học {query} site:vn",
            "api_key": api_key,
            "hl": "vi",
            "gl": "vn",
            "num": max_results,
        }
        fb_results, fb_error = _run_google_search(params_fallback)

        if not fb_error:
            for item in fb_results:
                link = item.get("link", "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)

                collected_results.append({
                    "title": item.get("title", "Không có tiêu đề"),
                    "link": link,
                    "snippet": item.get("snippet", "Không có tóm tắt."),
                    "source": "Google / VN Research (mở rộng)",
                    "origin": "Tạp chí VN",
                })

    if not collected_results:
        return [], "Không tìm thấy bài báo tiếng Việt phù hợp. Bạn có thể thử đổi tên đề tài ngắn gọn hơn."

    return collected_results, None

# ============================================================
# 7. INGESTION TỪ API VÀO DATABASE
# ============================================================
def ingest_pubmed_article(article: Dict[str, Any]) -> bool:
    key = article.get("pmid") or article.get("url") or article["title"]
    file_hash = sha256_text(key)
    source_id = make_source_id(f"PubMed:{key}", file_hash)

    source = SourceDocument(
        source_id=source_id,
        file_name=article["title"][:120],
        file_hash=file_hash,
        origin="PubMed",
        title=article["title"],
        authors=article.get("authors", ""),
        year=article.get("year", ""),
        journal=article.get("journal", ""),
        doi=article.get("doi", ""),
        pmid=article.get("pmid", ""),
        url=article.get("url", ""),
    )

    chunks = []
    for idx, (piece, start, end) in enumerate(
        split_text_into_chunks(article.get("abstract", "")), start=1
    ):
        chunks.append(
            EvidenceChunk(
                chunk_id=make_chunk_id(source_id, 0, idx),
                source_id=source_id,
                file_name=source.file_name,
                page=0,
                text=piece,
                char_start=start,
                char_end=end,
                section="Abstract (PubMed)",
            )
        )
    return add_source_and_chunks(source, chunks)

def ingest_vn_article(article: Dict[str, Any]) -> bool:
    key = article.get("link") or article["title"]
    file_hash = sha256_text(key)
    source_id = make_source_id(f"VN:{key}", file_hash)

    source = SourceDocument(
        source_id=source_id,
        file_name=article["title"][:120],
        file_hash=file_hash,
        origin="Tạp chí VN",
        title=article["title"],
        journal=article.get("source", ""),
        url=article.get("link", ""),
    )

    chunks = []
    snippet = article.get("snippet", "")
    for idx, (piece, start, end) in enumerate(split_text_into_chunks(snippet), start=1):
        chunks.append(
            EvidenceChunk(
                chunk_id=make_chunk_id(source_id, 0, idx),
                source_id=source_id,
                file_name=source.file_name,
                page=0,
                text=piece,
                char_start=start,
                char_end=end,
                section="Đoạn trích (Google Scholar)",
                table_hint="CHỈ LÀ ĐOẠN TRÍCH NGẮN - CẦN KIỂM TRA BẢN GỐC TRƯỚC KHI DÙNG SỐ LIỆU",
            )
        )
    return add_source_and_chunks(source, chunks)
