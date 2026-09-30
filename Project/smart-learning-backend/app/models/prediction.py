import uuid
from decimal import Decimal
from typing import Optional

from sqlalchemy import ForeignKey, Numeric, String, UUID
from sqlalchemy.dialects.postgresql import BIGINT, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Prediction(Base):
    __tablename__ = "pass_predictions"

    prediction_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(BIGINT, ForeignKey("model_registry.model_id"), nullable=False)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("student_profiles.student_id"), nullable=False)
    course_id: Mapped[int] = mapped_column(BIGINT, ForeignKey("courses.course_id"), nullable=False)
    teacher_id: Mapped[int] = mapped_column(BIGINT, ForeignKey("teachers.teacher_id"), nullable=False)
    offering_id: Mapped[Optional[int]] = mapped_column(BIGINT, ForeignKey("course_teacher_offerings.offering_id"), nullable=True)
    predicted_pass_probability: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    risk_band: Mapped[str] = mapped_column(String(20), nullable=False)
    top_factors: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
