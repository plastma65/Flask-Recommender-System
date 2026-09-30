import uuid
from decimal import Decimal
from typing import Optional

from sqlalchemy import String, Boolean, SmallInteger, ForeignKey, Numeric, UUID
from sqlalchemy.dialects.postgresql import BIGINT, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base  # Dùng chung Base duy nhất để Alembic nhận đủ bảng


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_type: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    training_dataset_snapshot: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    evaluation_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class RecommendationResult(Base):
    __tablename__ = "recommendation_results"

    recommendation_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(BIGINT, ForeignKey("model_registry.model_id"), nullable=False)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("student_profiles.student_id"), nullable=False)
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)
    course_id: Mapped[Optional[int]] = mapped_column(BIGINT, ForeignKey("courses.course_id"), nullable=True)
    teacher_id: Mapped[Optional[int]] = mapped_column(BIGINT, ForeignKey("teachers.teacher_id"), nullable=True)
    offering_id: Mapped[Optional[int]] = mapped_column(BIGINT, ForeignKey("course_teacher_offerings.offering_id"), nullable=True)
    recommendation_rank: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    recommendation_score: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    explanation_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
