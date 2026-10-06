import time

def check_internet_plagiarism(text: str, max_queries: int = 3) -> list:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("Vui lòng cài đặt thư viện: pip install duckduckgo-search")

    clean_text = text.replace('\n', ' ').strip()
    words = clean_text.split()
    
    if len(words) < 15:
        return [] 
        
    chunk_size = 15
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size) if len(words[i:i+chunk_size]) >= 10]
    
    if len(chunks) > max_queries:
        chunks_to_check = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
    else:
        chunks_to_check = chunks

    results = []
    
    with DDGS() as ddgs:
        for chunk in chunks_to_check:
            query = f'"{chunk}"' 
            try:
                search_results = list(ddgs.text(query, max_results=2, region='wt-wt', safesearch='off'))
                if search_results:
                    for res in search_results:
                        results.append({
                            "Đoạn văn gốc": chunk + "...",
                            "Tiêu đề trang": res.get("title", "Không rõ"),
                            "Nguồn (URL)": res.get("href", ""),
                            "Đoạn trích (Snippet)": res.get("body", "")
                        })
            except Exception as e:
                pass
            time.sleep(1) 
            
    return results
