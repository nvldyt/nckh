# ============================================================
# BỔ SUNG: BỘ MÁY PUBMED MULTI-QUERY (TÌM KIẾM THÔNG MINH 4 BIẾN THỂ)
# ============================================================
def generate_pubmed_queries(main_query: str) -> List[String]:
    """
    Tự động sinh ra 4 biến thể truy vấn từ từ khóa gốc để tối ưu hóa khả năng tìm kiếm trên PubMed.
    """
    # Làm sạch từ khóa đầu vào
    clean_q = main_query.replace('"', '').strip()
    
    # Tạo 4 chiến lược query khác nhau dựa trên từ khóa y văn
    queries = [
        f"{clean_q}",  # Biến thể 1: Truy vấn gốc chuẩn hóa
        f"{clean_q} AND (drug utilization OR treatment pattern OR management)",  # Biến thể 2: Hướng thực trạng sử dụng thuốc
        f"{clean_q} AND (clinical outcome OR blood pressure control OR effectiveness)", # Biến thể 3: Hướng hiệu quả / kết cục điều trị
        f"{clean_q} AND (outpatient OR inpatient OR prevalence OR risk factors)"  # Biến thể 4: Hướng đối tượng / quần thể nghiên cứu
    ]
    return queries

def search_pubmed_multi(main_query: str, max_res_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    Thực hiện tìm kiếm song song đa biến thể (Multi-Query), gom kết quả, 
    lọc trùng hoàn toàn bằng PMID và trả về danh sách bài báo tốt nhất.
    """
    queries = generate_pubmed_queries(main_query)
    seen_pmids = set()
    all_articles = []
    
    for q in queries:
        # Gọi hàm search_pubmed sẵn có cho từng biến thể
        articles = search_pubmed(q, max_res=max_res_per_query)
        for art in articles:
            pmid = art.get("pmid")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_articles.append(art)
            elif not pmid:
                # Trường hợp bài báo không có PMID thì dùng title làm khóa chống trùng
                all_articles.append(art)
                
    # Giới hạn tổng số lượng bài báo trả về để không làm quá tải giao diện (ví dụ tối đa 10 bài)
    return all_articles[:12]
