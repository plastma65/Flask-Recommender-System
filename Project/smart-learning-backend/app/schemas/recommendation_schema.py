from typing import List

from pydantic import BaseModel, Field


# ── Teacher Recommendation Schemas ──────────────────────────────────────


class TeacherRecommendationRequest(BaseModel):
    student_id: str | int | None = None
    query_text: str = Field(
        ..., min_length=1, max_length=500,
        description="Mô tả nhu cầu tìm giảng viên bằng tiếng Việt",
    )
    alpha: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Trọng số Content-Based (0→chỉ SVD, 1→chỉ NLP)",
    )
    top_k: int = Field(default=5, ge=1, le=10)


class TeacherRecommendationItem(BaseModel):
    teacher_id: int
    teacher_name: str = ""
    content_score: float
    collab_score: float
    hybrid_score: float
    course_name: str = ""


class TeacherRecommendationResponseData(BaseModel):
    items: List[TeacherRecommendationItem]


# ── Course Recommendation Schemas ───────────────────────────────────────


class CourseRecommendationRequest(BaseModel):
    semester: str
    target_credit_load: int
    top_k: int = Field(default=5, ge=1, le=10)


class CourseRecommendationItem(BaseModel):
    course_id: int
    course_name: str
    fit_score: float
    expected_difficulty: int = Field(..., ge=1, le=5)
    reasons: List[str]


class CourseRecommendationResponseData(BaseModel):
    items: List[CourseRecommendationItem]
