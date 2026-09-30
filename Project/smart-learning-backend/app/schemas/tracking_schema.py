from typing import Optional

from pydantic import BaseModel, Field


class LogClickRequest(BaseModel):
    """Ghi nhận hành vi người dùng với một gợi ý (xem / so sánh / chọn)."""
    recommendation_context_id: int
    teacher_id: Optional[int] = None
    course_id: Optional[int] = None
    action_type: str = Field(..., pattern="^(VIEW|COMPARE|SELECT)$")


class PostRecoFeedbackRequest(BaseModel):
    """Phản hồi sau khi sinh viên sử dụng gợi ý."""
    recommendation_context_id: int
    used: bool
    satisfaction_score: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
