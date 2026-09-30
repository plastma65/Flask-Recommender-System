import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.reco_service import get_teacher_recommendations, get_course_recommendations

from app.core.database import get_db
from app.schemas.response import APIResponse, ResponseMeta
from app.schemas.recommendation_schema import (
    TeacherRecommendationRequest,
    TeacherRecommendationResponseData,
    TeacherRecommendationItem,
    CourseRecommendationRequest,
    CourseRecommendationResponseData,
    CourseRecommendationItem,
)
from app.schemas.tracking_schema import (
    LogClickRequest,
    PostRecoFeedbackRequest,
)

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────


def _meta() -> ResponseMeta:
    """Tạo metadata chuẩn cho mọi response."""
    return ResponseMeta(
        request_id=uuid.uuid4().hex,
        timestamp=datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
    )


# ── Teacher Recommendation ──────────────────────────────────────────────


@router.post("/teachers", response_model=APIResponse[TeacherRecommendationResponseData])
async def recommend_teachers_endpoint(
    request: TeacherRecommendationRequest,
) -> APIResponse[TeacherRecommendationResponseData]:

    raw_items = await get_teacher_recommendations(
        student_id=request.student_id,
        query_text=request.query_text,
        alpha=request.alpha,
        top_k=request.top_k,
    )

    parsed_items = [TeacherRecommendationItem(**item) for item in raw_items]

    return APIResponse(
        success=True,
        message="Teacher recommendations retrieved successfully",
        data=TeacherRecommendationResponseData(items=parsed_items),
        meta=_meta(),
    )


# ── Course Recommendation ───────────────────────────────────────────────


@router.post("/courses", response_model=APIResponse[CourseRecommendationResponseData])
async def recommend_courses_endpoint(
    request: CourseRecommendationRequest,
    session: AsyncSession = Depends(get_db),
) -> APIResponse[CourseRecommendationResponseData]:

    raw_items = await get_course_recommendations(
        session=session,
        student_id="mock_student_uuid",  # TODO: lấy từ JWT token
        semester=request.semester,
        target_credit_load=request.target_credit_load,
        top_k=request.top_k,
    )

    items = [CourseRecommendationItem(**item) for item in raw_items]

    return APIResponse(
        success=True,
        message="Course recommendations retrieved successfully",
        data=CourseRecommendationResponseData(items=items),
        meta=_meta(),
    )


# ── Tracking ────────────────────────────────────────────────────────────


@router.post("/log-click", response_model=APIResponse[dict])
async def log_click(
    request: LogClickRequest,
    session: AsyncSession = Depends(get_db),
) -> APIResponse[dict]:

    return APIResponse(
        success=True,
        message="Click logged successfully",
        data={"status": "ok"},
        meta=_meta(),
    )


@router.post("/post-feedback", response_model=APIResponse[dict])
async def post_recommendation_feedback(
    request: PostRecoFeedbackRequest,
    session: AsyncSession = Depends(get_db),
) -> APIResponse[dict]:

    return APIResponse(
        success=True,
        message="Recommendation feedback saved",
        data={"saved": True},
        meta=_meta(),
    )
