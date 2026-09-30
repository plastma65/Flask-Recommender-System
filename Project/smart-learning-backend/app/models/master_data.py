import uuid

from sqlalchemy import String, SmallInteger, Boolean, UUID
from sqlalchemy.dialects.postgresql import BIGINT
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Course(Base):
    __tablename__ = "courses"

    course_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    course_code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    course_name: Mapped[str] = mapped_column(String(200), nullable=False)
    credit_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)


class Teacher(Base):
    __tablename__ = "teachers"

    teacher_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    teacher_name: Mapped[str] = mapped_column(String(150), nullable=False)
    department_name: Mapped[str] = mapped_column(String(150), nullable=False)


class StudentProfile(Base):
    """Hồ sơ học tập sinh viên – các cột khớp 1-1 với student_profiles.csv."""
    __tablename__ = "student_profiles"

    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gpa_bucket: Mapped[str] = mapped_column(String(30), nullable=False)
    failed_subjects_count: Mapped[str] = mapped_column(String(30), nullable=False)
    study_hours_bucket: Mapped[str] = mapped_column(String(30), nullable=False)
    preferred_study_time: Mapped[str] = mapped_column(String(20), nullable=False)
    preferred_class_style: Mapped[str] = mapped_column(String(20), nullable=False)
    learning_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    study_group_pref: Mapped[str] = mapped_column(String(30), nullable=False)
