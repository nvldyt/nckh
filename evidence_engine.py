# evidence_engine.py

import io
import re
import os
import hashlib
from serpapi import GoogleSearch
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import requests
import xml.etree.ElementTree as ET
import streamlit as st
from pypdf import PdfReader

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
def split_text_into_chunks(
    text: str, chunk_size: int = 1800, overlap: int = 300
) -> List[Tuple[str, int, int]]:
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
# 5. XỬ LÝ PDF
# ============================================================
def extract_pdf(uploaded_file) -> Tuple[SourceDocument, List[EvidenceChunk]]:
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

# ============================================================
# 6. TRA CỨU API (PUBMED & VN JOURNALS)
# ============================================================
def get_serpapi_key() -> Optional[str]:
    try:
        return st.secrets.get("SERPAPI_KEY", "")
    except Exception:
        return os.getenv("SERPAPI_KEY", "")

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

def search_vn_journals(query: str, max_results: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Tìm kiếm bài báo trên các tạp chí y học Việt Nam thông qua SerpAPI.
    Đã bổ sung cơ chế dự phòng (fallback) nếu từ khóa rút gọn quá ngắn không ra kết quả.
    """
    api_key = get_serpapi_key()
    if not api_key:
        return [], "⚠️ Chưa cấu hình SerpAPI Key (không thể cào kết quả từ tạp chí VN)."

    domains = st.session_state.get("vn_journal_domains", [])
    if not domains:
        domains = ["tapchiyhocvietnam.vn", "vjol.info", "tapchinghiencuuyhoc.vn"]

    # Tạo câu lệnh tìm kiếm giới hạn trong các trang tạp chí y học VN
    site_query = " OR ".join([f"site:{d}" for d in domains])
    
    # Danh sách các biến thể từ khóa để thử tìm kiếm lần lượt nếu lần đầu bị rỗng
    search_queries = [
        f"{query} (sử dụng OR vancomycin OR điều trị)", # Kết hợp từ khóa người dùng
        query,                                         # Từ khóa thô
    ]
    
    # Nếu query quá ngắn (ví dụ chỉ có 1 từ như "Vancomycin"), ép tìm kèm chữ y học
    if len(query.split()) <= 1:
        search_queries.insert(0, f"{query} dược lâm sàng bệnh viện")

    collected_results = []
    seen_links = set()

    for q_text in search_queries:
        if len(collected_results) >= max_results:
            break
            
        full_query = f"({site_query}) {q_text}"
        
        params = {
            "engine": "google",
            "q": full_query,
            "api_key": api_key,
            "hl": "vi",
            "gl": "vn",
            "num": max_results
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])

            for item in organic_results:
                link = item.get("link", "")
                if link in seen_links:
                    continue
                seen_links.add(link)

                title = item.get("title", "Không có tiêu đề")
                snippet = item.get("snippet", "Không có tóm tắt.")
                
                # Trích xuất tên nguồn từ link hiển thị hoặc domain
                source_name = "Tạp chí Y học Việt Nam"
                for d in domains:
                    if d in link:
                        source_name = d.upper()
                        break

                collected_results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "source": source_name,
                    "origin": "Tạp chí VN"
                })
                
                if len(collected_results) >= max_results:
                    break
        except Exception as e:
            return [], f"Lỗi kết nối SerpAPI: {str(e)}"

    if not collected_results:
        return [], "Không tìm thấy bài báo tiếng Việt phù hợp với từ khóa này. Bạn có thể thử đổi tên đề tài ngắn gọn hơn ở ô tra cứu."

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
