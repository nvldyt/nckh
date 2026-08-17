import os
import json
import pandas as pd
import streamlit as st
from dataclasses import asdict
from table_selection_engine import CandidateResult

PROJECT_DIR = "project_data"

def save_project(project_name: str):
    if not project_name.strip():
        return False, "Tên dự án không được để trống."
    
    proj_path = os.path.join(PROJECT_DIR, project_name.strip())
    os.makedirs(proj_path, exist_ok=True)
    
    try:
        # 1. Lưu metadata & citation registry
        meta_data = {
            "documents": st.session_state.get("documents", {}),
            "citation_registry": st.session_state.get("citation_registry", {}),
        }
        with open(os.path.join(proj_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
        
        # 2. Lưu chunks văn bản bằng chứng
        chunks = st.session_state.get("chunks", [])
        with open(os.path.join(proj_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)

        # 3. Lưu result_cart (Chuyển đổi CandidateResult dataclass sang dict)
        cart = st.session_state.get("result_cart", [])
        cart_dicts = [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in cart]
        with open(os.path.join(proj_path, "result_cart.json"), "w", encoding="utf-8") as f:
            json.dump(cart_dicts, f, ensure_ascii=False, indent=4)

        # 4. Lưu saved_tables (Các bảng DataFrame thành định dạng parquet tối ưu)
        saved_tables = st.session_state.get("saved_tables", {})
        tables_dir = os.path.join(proj_path, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        for rid, df in saved_tables.items():
            if isinstance(df, pd.DataFrame):
                df.to_parquet(os.path.join(tables_dir, f"{rid}.parquet"))

        return True, f"Đã lưu checkpoint dự án '{project_name}' thành công!"
    except Exception as e:
        return False, f"Lỗi khi lưu dự án: {e}"

def load_project(project_name: str):
    proj_path = os.path.join(PROJECT_DIR, project_name)
    if not os.path.exists(proj_path):
        return False, "Không tìm thấy thư mục dự án."
    
    try:
        # 1. Khôi phục metadata
        meta_path = os.path.join(proj_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                st.session_state["documents"] = meta.get("documents", {})
                st.session_state["citation_registry"] = meta.get("citation_registry", {})

        # 2. Khôi phục chunks
        chunks_path = os.path.join(proj_path, "chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                st.session_state["chunks"] = json.load(f)

        # 3. Khôi phục result_cart
        cart_path = os.path.join(proj_path, "result_cart.json")
        if os.path.exists(cart_path):
            with open(cart_path, "r", encoding="utf-8") as f:
                cart_dicts = json.load(f)
                st.session_state["result_cart"] = [CandidateResult(**item) for item in cart_dicts]

        # 4. Khôi phục saved_tables
        tables_dir = os.path.join(proj_path, "tables")
        if os.path.exists(tables_dir):
            saved_tables = {}
            for file in os.listdir(tables_dir):
                if file.endswith(".parquet"):
                    rid = file[:-8] # Cắt bỏ đuôi .parquet
                    saved_tables[rid] = pd.read_parquet(os.path.join(tables_dir, file))
            st.session_state["saved_tables"] = saved_tables

        # Dựng lại index vector embedding sau khi load
        from evidence_engine import rebuild_index
        rebuild_index()

        return True, f"Đã khôi phục thành công dự án '{project_name}'!"
    except Exception as e:
        return False, f"Lỗi khi tải dự án: {e}"

def list_projects():
    if not os.path.exists(PROJECT_DIR):
        return []
    return [d for d in os.listdir(PROJECT_DIR) if os.path.isdir(os.path.join(PROJECT_DIR, d))]
