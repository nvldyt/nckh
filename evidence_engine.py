# File: evidence_engine.py (Bản TỐI GIẢN - Chuyên xử lý PubMed & PDF, Tối đa 20 bài)

import io
import re
import os
import hashlib
import concurrent.futures
import gc  # Thêm thư viện dọn rác RAM
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from semantic_chunker import SemanticChunker

import requests
import xml.etree.ElementTree as ET
import streamlit as st
import fitz
from openai import OpenAI # BỔ SUNG THƯ VIỆN ĐỂ GỌI TRỰC TIẾP GROQ

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
        chunker = SemanticChunker(max_chunk_size=1200, min_chunk_size=100, chunk_overlap=250)
        
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
        del data
        gc.collect()

    return source, chunks
    
# ============================================================
# 6. TRA CỨU API (PUBMED QUỐC TẾ - MỞ RỘNG 20 BÀI)
# ============================================================
def translate_query_to_mesh(vietnamese_topic: str) -> str:
    """Tích hợp trực tiếp dịch thuật AI vào engine để không phụ thuộc file api.py"""
    groq_key = os.getenv("GROQ_API_KEYS", "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")
    
    system_prompt = (
        "Bạn là một Chuyên gia Thư viện Y khoa (Medical Librarian) chuyên tra cứu PubMed.\n"
        "Nhiệm vụ: Dịch đề tài nghiên cứu tiếng Việt sang cú pháp tìm kiếm MeSH của PubMed.\n\n"
        "Quy tắc tuyệt đối:\n"
        "- CHỈ trả về chuỗi boolean query cuối cùng.\n"
        "- TUYỆT ĐỐI KHÔNG giải thích, KHÔNG bọc ngoặc kép ở 2 đầu chuỗi.\n\n"
        "--- TỪ ĐIỂN ÁNH XẠ MESH BẮT BUỘC ---\n"
        "- Thuốc, dược phẩm: \"Pharmaceutical Preparations\"[Mesh] OR \"Drug Therapy\"[Mesh] OR \"Drug Utilization\"[Mesh]\n"
        "- Kháng sinh: \"Anti-Bacterial Agents\"[Mesh]\n"
        "- Bệnh đái đường / đái tháo đường: \"Diabetes Mellitus\"[Mesh]\n"
        "- Bệnh đau dạ dày: \"Stomach Diseases\"[Mesh] OR \"Gastritis\"[Mesh]\n"
        "- Bệnh viêm gan: \"Hepatitis\"[Mesh]\n"
        "- Bệnh đau tim / Nhồi máu cơ tim: \"Heart Diseases\"[Mesh] OR \"Myocardial Infarction\"[Mesh]\n"
        "------------------------------------\n\n"
        "--- VÍ DỤ CHUẨN (FEW-SHOT) ---\n"
        "Input: Phân tích tình hình sử dụng thuốc đái tháo đường\n"
        "Output: \"Diabetes Mellitus\"[Mesh] AND (\"Drug Utilization\"[Mesh] OR \"Drug Therapy\"[Mesh])\n\n"
        "Input: Đánh giá hiệu quả điều trị bệnh đau dạ dày bằng kháng sinh\n"
        "Output: (\"Stomach Diseases\"[Mesh] OR \"Gastritis\"[Mesh]) AND \"Anti-Bacterial Agents\"[Mesh] AND \"Treatment Outcome\"[Mesh]"
    )
    
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1", 
            api_key=groq_key, 
            timeout=15.0
        )
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input: {vietnamese_topic}\nOutput:"}
            ],
            temperature=0.0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip().strip('"').strip("'")
        
    except Exception as e:
        return f"Lỗi gọi AI: {str(e)}"

def search_pubmed(query_en: str, max_res: int = 20) -> List[Dict[str, Any]]:
    """Đã chỉnh sửa max_res mặc định lên 20"""
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
    clean_q = main_query.replace('"', '').strip()
    return [
        f"{clean_q}",
        f"{clean_q} AND (drug utilization OR treatment pattern OR management)",
        f"{clean_q} AND (clinical outcome OR blood pressure control OR effectiveness)",
        f"{clean_q} AND (outpatient OR inpatient OR prevalence OR risk factors)"
    ]

def search_pubmed_multi(main_query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Hàm xử lý đa luồng, tham số truyền vào max_results mặc định là 20"""
    queries = generate_pubmed_queries(main_query)
    seen_pmids = set()
    all_articles = []
    
    # Tính số bài cần tìm cho mỗi luồng (query) để gom đủ tổng số bài
    max_res_per_query = max(5, (max_results // len(queries)) + 3)
    
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
                
    # Trả về đúng số lượng bài theo yêu cầu trên giao diện
    return all_articles[:max_results]

# ============================================================
# 7. INGESTION TỪ API VÀO DATABASE (CHỈ CÒN PUBMED)
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
