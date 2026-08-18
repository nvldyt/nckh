# project_storage.py
# ============================================================
# QUẢN LÝ CHECKPOINT / DỰ ÁN LUẬN VĂN (LƯU CỤC BỘ XUỐNG ĐĨA)
# ============================================================
#
# Module này cho phép lưu lại toàn bộ trạng thái làm việc hiện tại
# (Evidence Database, Citation Registry, Result Database của Tab 4,
# cấu trúc Chương 3 đã tuyển chọn, bản nháp gần nhất...) xuống một
# file .pkl trên đĩa của máy chủ, để có thể khôi phục lại sau khi
# app bị restart / mất phiên Streamlit.
#
# Lưu ý: đây là lưu trữ CỤC BỘ theo máy chủ đang chạy app. Nếu deploy
# trên Streamlit Community Cloud, thư mục này có thể bị xóa khi app
# "ngủ đông" / redeploy — nên xem đây là checkpoint tạm thời trong
# phiên làm việc, không phải nơi lưu trữ vĩnh viễn duy nhất.

import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Thư mục lưu các file checkpoint dự án (sử dụng pathlib hiện đại)
PROJECTS_DIR = Path.cwd() / "saved_projects"

# Các key trong st.session_state sẽ được đóng gói khi lưu 1 dự án.
PROJECT_STATE_KEYS = [
    # Evidence Database (Tab 1 + Tab 2)
    "documents",
    "chunks",
    "embeddings",
    "citation_registry",
    "vn_journal_domains",
    "t3_pm_data",
    "t3_vn_data",
    "t3_en_keyword",
    "t3_query",
    # Viết luận văn (Tab 3)
    "last_generated",
    "last_evidence",
    "audit_log",
    # Phân tích số liệu & cấu trúc Chương 3 (Tab 4)
    "result_cart",
    "saved_tables",
    "selection_decisions",
    "narrative_plan",
    "structure_locked",
]


def _ensure_dir() -> None:
    """Đảm bảo thư mục lưu trữ luôn tồn tại."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    """Chuẩn hoá tên dự án thành tên file an toàn (bỏ ký tự đặc biệt)."""
    name = (name or "").strip() or "du_an_khong_ten"
    keep_chars = "-_.() "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep_chars).strip()
    return cleaned.replace(" ", "_") or "du_an_khong_ten"


def save_project(project_name: str) -> Tuple[bool, str]:
    """Lưu snapshot toàn bộ trạng thái nghiên cứu hiện tại xuống đĩa."""
    try:
        _ensure_dir()
        filename = _safe_filename(project_name)
        filepath = PROJECTS_DIR / f"{filename}.pkl"

        # Đóng gói dữ liệu bằng dict comprehension để mã gọn và nhanh hơn
        snapshot = {
            key: st.session_state[key]
            for key in PROJECT_STATE_KEYS
            if key in st.session_state
        }

        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot["_saved_at"] = saved_at
        snapshot["_project_name"] = project_name

        with filepath.open("wb") as f:
            pickle.dump(snapshot, f)

        n_keys = len(snapshot) - 2  # trừ 2 key metadata
        return True, (
            f"✅ Đã lưu checkpoint dự án **'{project_name}'** lúc {saved_at} "
            f"({n_keys} nhóm dữ liệu đã đóng gói)."
        )
    except Exception as exc:
        return False, f"❌ Lỗi khi lưu dự án: {exc}"


def load_project(project_name: str) -> Tuple[bool, str]:
    """Khôi phục lại snapshot đã lưu vào st.session_state hiện tại."""
    try:
        _ensure_dir()
        filename = _safe_filename(project_name)
        filepath = PROJECTS_DIR / f"{filename}.pkl"

        if not filepath.exists():
            # project_name có thể đã là tên file gốc, thử tìm trực tiếp
            alt_path = PROJECTS_DIR / project_name
            if alt_path.suffix != ".pkl":
                alt_path = alt_path.with_suffix(".pkl")
            
            if alt_path.exists():
                filepath = alt_path
            else:
                return False, f"❌ Không tìm thấy checkpoint '{project_name}'."

        with filepath.open("rb") as f:
            snapshot = pickle.load(f)

        # Nạp lại dữ liệu vào session state
        for key in PROJECT_STATE_KEYS:
            if key in snapshot:
                st.session_state[key] = snapshot[key]

        saved_at = snapshot.get("_saved_at", "không rõ thời điểm")
        return True, f"✅ Đã khôi phục dự án **'{project_name}'** (lưu lúc {saved_at})."
    except Exception as exc:
        return False, f"❌ Lỗi khi tải dự án: {exc}"


def list_projects() -> List[str]:
    """Liệt kê các checkpoint đã lưu, mới nhất lên trước."""
    try:
        _ensure_dir()
        # Lấy tất cả file .pkl và sắp xếp theo thời gian sửa đổi (mới nhất đầu tiên)
        files = list(PROJECTS_DIR.glob("*.pkl"))
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        # Trả về tên file bỏ đuôi .pkl (dùng f.stem)
        return [f.stem for f in files]
    except Exception:
        return []


def delete_project(project_name: str) -> Tuple[bool, str]:
    """Xóa 1 checkpoint dự án khỏi đĩa (tuỳ chọn, dùng khi cần dọn dẹp)."""
    try:
        _ensure_dir()
        filename = _safe_filename(project_name)
        filepath = PROJECTS_DIR / f"{filename}.pkl"
        
        if filepath.exists():
            filepath.unlink()
            return True, f"🗑️ Đã xóa checkpoint '{project_name}'."
        return False, f"❌ Không tìm thấy checkpoint '{project_name}' để xóa."
    except Exception as exc:
        return False, f"❌ Lỗi khi xóa dự án: {exc}"
