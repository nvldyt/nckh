from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import json
import re
from pathlib import Path


class Priority(str, Enum):
    CORE = "CORE"
    SUPPORTING = "SUPPORTING"
    OPTIONAL = "OPTIONAL"
    APPENDIX = "APPENDIX"


class Presentation(str, Enum):
    TABLE = "TABLE"
    TEXT = "TEXT"
    FIGURE = "FIGURE"
    APPENDIX = "APPENDIX"


@dataclass
class StudyObjective:
    id: str
    title: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)


@dataclass
class CandidateResult:
    id: str
    title: str
    result_type: str = "demographic"
    variables: List[str] = field(default_factory=list)
    objective_ids: List[str] = field(default_factory=list)

    n: Optional[int] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    clinical_importance: float = 0.0
    discussion_value: float = 0.0
    scientific_value: float = 0.0
    complexity: float = 0.0

    parent_result_id: Optional[str] = None

    locked_priority: Optional[Priority] = None
    locked_presentation: Optional[Presentation] = None
    locked_order: Optional[int] = None

    notes: str = ""
    source_ids: List[str] = field(default_factory=list)
    chapter_hint: Optional[str] = None


@dataclass
class SelectionDecision:
    result_id: str
    title: str
    objective_ids: List[str]
    objective_score: float
    scientific_score: float
    clinical_score: float
    discussion_score: float
    independence_score: float
    statistical_score: float
    total_score: float
    priority: Priority
    presentation: Presentation
    recommended_order: Optional[int] = None
    duplicate_with: List[str] = field(default_factory=list)
    parent_result_id: Optional[str] = None
    reason: str = ""
    reviewer_note: str = ""
    locked: bool = False


_STOPWORDS = {
    "và", "của", "cho", "trên", "theo", "với", "trong", "là", "có",
    "được", "nhóm", "tỷ", "lệ", "phân", "bệnh", "nhân", "đối", "tượng",
    "nghiên", "cứu", "bảng", "đặc", "điểm", "tình", "hình"
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return {
        t for t in normalize_text(text).split()
        if t and t not in _STOPWORDS
    }


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def result_similarity(a: CandidateResult, b: CandidateResult) -> float:
    title_sim = jaccard(tokens(a.title), tokens(b.title))
    var_sim = jaccard(
        [normalize_text(x) for x in a.variables],
        [normalize_text(x) for x in b.variables],
    )
    obj_sim = jaccard(a.objective_ids, b.objective_ids)
    return 0.50 * title_sim + 0.40 * var_sim + 0.10 * obj_sim


class TableSelectionEngine:
    """Chọn, xếp hạng và sắp xếp kết quả cho Chương Kết quả theo chuẩn luận văn Y học."""

    def __init__(
        self,
        objectives: Sequence[StudyObjective],
        candidates: Sequence[CandidateResult],
        duplicate_threshold: float = 0.68,
    ):
        self.objectives = list(objectives)
        self.candidates = list(candidates)
        self.duplicate_threshold = duplicate_threshold
        self._objective_map = {o.id: o for o in self.objectives}

    def infer_objectives(self, result: CandidateResult) -> List[str]:
        if result.objective_ids:
            return result.objective_ids

        result_tokens = tokens(
            " ".join([result.title, *result.variables, result.notes])
        )
        scored = []

        for obj in self.objectives:
            obj_tokens = tokens(
                " ".join([obj.title, obj.description, *obj.keywords])
            )
            scored.append((obj.id, jaccard(result_tokens, obj_tokens)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [scored[0][0]] if scored and scored[0][1] >= 0.12 else []

    def objective_score(self, result: CandidateResult) -> float:
        if result.objective_ids:
            return 5.0

        ids = self.infer_objectives(result)
        if not ids:
            return 0.0

        result_tokens = tokens(
            " ".join([result.title, *result.variables])
        )
        best = 0.0

        for oid in ids:
            obj = self._objective_map.get(oid)
            if obj:
                obj_tokens = tokens(
                    " ".join([obj.title, obj.description, *obj.keywords])
                )
                best = max(best, 5.0 * jaccard(result_tokens, obj_tokens))

        return round(min(best, 5.0), 2)

    def statistical_score(self, result: CandidateResult) -> float:
        score = 0.0
        if result.effect_size is not None:
            score += 1.5
        if result.ci_low is not None and result.ci_high is not None:
            score += 1.0
        if result.p_value is not None:
            if result.p_value < 0.05:
                score += 0.5
            elif result.p_value < 0.10:
                score += 0.2
        return min(round(score, 2), 3.0)

    def calculate_independence_scores(self) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """
        Thuật toán Alpha-Clustering: Giải quyết triệt để hiện tượng 'Cùng nhau chìm xuồng'.
        Bảng chất lượng tốt hơn (Bản gốc/Alpha) giữ nguyên 5.0 điểm, chỉ phạt các bản sao yếu hơn.
        """
        base_quality = {}
        for res in self.candidates:
            q_score = (
                res.scientific_value + 
                res.clinical_importance + 
                res.discussion_value + 
                self.statistical_score(res)
            )
            base_quality[res.id] = q_score

        # Sắp xếp từ bảng xuất sắc nhất đến kém nhất
        sorted_cands = sorted(self.candidates, key=lambda x: base_quality[x.id], reverse=True)
        
        independence_scores = {c.id: 5.0 for c in self.candidates}
        duplicates_map = {c.id: [] for c in self.candidates}
        seen_alphas = []

        for c in sorted_cands:
            if c.parent_result_id:
                independence_scores[c.id] = 3.0
                continue
                
            is_duplicate = False
            for alpha in seen_alphas:
                if result_similarity(c, alpha) >= self.duplicate_threshold:
                    # Chỉ phạt bản sao, giữ nguyên điểm bản gốc (Alpha)
                    independence_scores[c.id] = max(1.0, independence_scores[c.id] - 2.0)
                    duplicates_map[c.id].append(alpha.id)
                    is_duplicate = True
            
            if not is_duplicate:
                seen_alphas.append(c)

        return independence_scores, duplicates_map

    def total_score(
        self,
        result: CandidateResult,
        independence_score: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Tính điểm tổng hợp đa tiêu chí (MCDA) dựa trên điểm độc lập đã phân rã."""
        scores = {
            "objective": self.objective_score(result),
            "scientific": min(max(result.scientific_value, 0), 5),
            "clinical": min(max(result.clinical_importance, 0), 5),
            "discussion": min(max(result.discussion_value, 0), 5),
            "independence": independence_score,
            "statistical": self.statistical_score(result),
        }

        total = (
            scores["objective"] * 0.30 +
            scores["scientific"] * 0.20 +
            scores["clinical"] * 0.20 +
            scores["discussion"] * 0.15 +
            scores["independence"] * 0.05 +
            (scores["statistical"] / 3.0 * 5.0) * 0.10
        )

        return round(total * 20.0, 2), scores

    def classify(
        self,
        result: CandidateResult,
        total: float,
        independence: float,
    ) -> Priority:
        if result.locked_priority:
            return result.locked_priority

        if result.parent_result_id and independence <= 3.0:
            return Priority.APPENDIX

        if total >= 80:
            return Priority.CORE
        if total >= 65:
            return Priority.SUPPORTING
        if total >= 50:
            return Priority.OPTIONAL
        return Priority.APPENDIX

    def presentation(
        self,
        result: CandidateResult,
        priority: Priority,
    ) -> Presentation:
        if result.locked_presentation:
            return result.locked_presentation

        if result.result_type.lower() in {
            "figure", "chart", "trend", "distribution"
        }:
            return Presentation.FIGURE

        if len(result.variables) <= 1 and result.complexity <= 1.5:
            if priority in {Priority.OPTIONAL, Priority.APPENDIX}:
                return Presentation.TEXT

        if priority == Priority.APPENDIX:
            return Presentation.APPENDIX

        return Presentation.TABLE

    def make_reason(
        self,
        result: CandidateResult,
        decision: SelectionDecision,
    ) -> str:
        parts = []

        if decision.objective_score >= 4:
            parts.append("trả lời trực tiếp mục tiêu nghiên cứu")
        elif decision.objective_score >= 2:
            parts.append("có liên quan đến mục tiêu nghiên cứu")
        else:
            parts.append("chưa chứng minh được liên quan trực tiếp đến mục tiêu")

        if decision.clinical_score >= 4:
            parts.append("có giá trị lâm sàng cao")
        if decision.discussion_score >= 4:
            parts.append("có tiềm năng bàn luận")

        if decision.duplicate_with:
            parts.append(
                "có kết quả tương tự với " +
                ", ".join(decision.duplicate_with)
            )

        if result.p_value is not None and result.p_value < 0.05:
            parts.append("có tín hiệu thống kê đáng chú ý (p<0,05)")

        if decision.presentation == Presentation.TEXT:
            parts.append("phù hợp trình bày bằng đoạn văn")
        elif decision.presentation == Presentation.APPENDIX:
            parts.append("phù hợp đưa vào phụ lục")

        return "; ".join(parts) + "."

    def recommended_key(
        self,
        result: CandidateResult,
        decision: SelectionDecision,
    ):
        objective_order = {
            obj.id: i for i, obj in enumerate(self.objectives)
        }
        obj_index = min(
            [objective_order.get(x, 999) for x in decision.objective_ids]
            or [999]
        )

        # Khớp loại kết quả linh hoạt bằng toán tử 'in' thay vì so khớp cứng nhắc
        res_type_lower = result.result_type.lower()
        type_weight = 50
        type_order_map = {
            "demographic": 1,
            "baseline": 2,
            "clinical": 3,
            "disease": 4,
            "treatment": 5,
            "drug": 6,
            "appropriateness": 7,
            "safety": 8,
            "interaction": 9,
            "outcome": 10,
            "association": 11,
            "regression": 12,
            "figure": 13,
        }
        for k, v in type_order_map.items():
            if k in res_type_lower:
                type_weight = v
                break

        return (
            obj_index,
            type_weight,
            -decision.total_score,
            result.title.lower(),
        )

    def run(self) -> List[SelectionDecision]:
        # Tích hợp thuật toán tính điểm độc lập thông minh
        ind_scores_map, duplicates_map = self.calculate_independence_scores()
        raw = []

        for result in self.candidates:
            objective_ids = self.infer_objectives(result)
            ind_score = ind_scores_map[result.id]
            total, scores = self.total_score(result, ind_score)
            priority = self.classify(
                result, total, scores["independence"]
            )
            presentation = self.presentation(result, priority)

            decision = SelectionDecision(
                result_id=result.id,
                title=result.title,
                objective_ids=objective_ids,
                objective_score=scores["objective"],
                scientific_score=scores["scientific"],
                clinical_score=scores["clinical"],
                discussion_score=scores["discussion"],
                independence_score=scores["independence"],
                statistical_score=scores["statistical"],
                total_score=total,
                priority=priority,
                presentation=presentation,
                duplicate_with=duplicates_map.get(result.id, []),
                parent_result_id=result.parent_result_id,
                locked=bool(
                    result.locked_priority
                    or result.locked_presentation
                    or result.locked_order is not None
                ),
            )

            decision.reason = self.make_reason(result, decision)
            raw.append((result, decision))

        raw.sort(key=lambda x: self.recommended_key(x[0], x[1]))

        order = 1
        for result, decision in raw:
            if result.locked_order is not None:
                decision.recommended_order = result.locked_order
            elif decision.priority != Priority.APPENDIX:
                decision.recommended_order = order
                order += 1

        rank = {
            Priority.CORE: 0,
            Priority.SUPPORTING: 1,
            Priority.OPTIONAL: 2,
            Priority.APPENDIX: 3,
        }

        raw.sort(
            key=lambda x: (
                rank[x[1].priority],
                x[1].recommended_order or 9999
            )
        )

        return [d for _, d in raw]


class NarrativePlanner:
    """Tạo cấu trúc mạch kể chuyện (Result Story) cho Chương Kết quả."""

    @staticmethod
    def build(
        decisions: Sequence[SelectionDecision],
    ) -> Dict[str, Any]:
        main = [
            d for d in decisions
            if d.priority in {Priority.CORE, Priority.SUPPORTING}
        ]
        optional = [
            d for d in decisions if d.priority == Priority.OPTIONAL
        ]
        appendix = [
            d for d in decisions if d.priority == Priority.APPENDIX
        ]

        by_objective: Dict[str, List[Dict[str, Any]]] = {}

        for d in main:
            for oid in d.objective_ids or ["UNASSIGNED"]:
                by_objective.setdefault(oid, []).append({
                    "order": d.recommended_order,
                    "result_id": d.result_id,
                    "title": d.title,
                    "priority": d.priority.value,
                    "presentation": d.presentation.value,
                    "score": d.total_score,
                    "reason": d.reason,
                })

        return {
            "main_results": [
                {
                    "order": d.recommended_order,
                    "result_id": d.result_id,
                    "title": d.title,
                    "presentation": d.presentation.value,
                    "priority": d.priority.value,
                    "score": d.total_score,
                }
                for d in main
            ],
            "optional_results": [
                {
                    "result_id": d.result_id,
                    "title": d.title,
                    "presentation": d.presentation.value,
                    "score": d.total_score,
                }
                for d in optional
            ],
            "appendix_results": [
                {
                    "result_id": d.result_id,
                    "title": d.title,
                    "presentation": d.presentation.value,
                    "score": d.total_score,
                }
                for d in appendix
            ],
            "by_objective": by_objective,
            "warnings": {
                "unassigned_results": [
                    d.result_id for d in decisions if not d.objective_ids
                ],
                "need_reviewer_confirmation": True,
            },
        }


def export_json(
    decisions: Sequence[SelectionDecision],
    path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            [asdict(d) for d in decisions],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def export_csv(
    decisions: Sequence[SelectionDecision],
    path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(d) for d in decisions]
    if not rows:
        p.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            r = dict(row)
            r["objective_ids"] = "; ".join(r["objective_ids"])
            r["duplicate_with"] = "; ".join(r["duplicate_with"])
            writer.writerow(r)
