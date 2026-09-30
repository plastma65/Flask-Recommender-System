import asyncio

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.lifespan import ml_models


async def get_teacher_recommendations(
    student_id: str | int | None,
    query_text: str,
    alpha: float = 0.6,
    top_k: int = 5,
) -> list[dict]:
    """
    Gợi ý giảng viên bằng thuật toán Hybrid (Content-Based + Collaborative Filtering).

    Lấy instance HybridTeacherRecommender đã nạp sẵn trong RAM từ lifespan,
    chạy inference trong thread pool để không block event loop.
    """
    recommender = ml_models.get("teacher_hybrid")
    if recommender is None:
        raise HTTPException(
            status_code=500,
            detail="Hybrid Recommender chưa sẵn sàng. Kiểm tra log khởi động server.",
        )

    results = await asyncio.to_thread(
        recommender.recommend,
        student_id=student_id,
        query_text=query_text,
        alpha=alpha,
        top_k=top_k,
    )
    return results


async def get_course_recommendations(
    session: AsyncSession,
    student_id: str,
    semester: str,
    target_credit_load: int,
    top_k: int = 5,
) -> list[dict]:
    """
    Gợi ý môn học nên đăng ký (Content-based).

    Hiện tại dùng dữ liệu mock; khi tích hợp model Content-based
    sẽ thay thế bằng logic tính fit_score thực tế.
    """

    mock_courses = [
        {
            "course_id": 1,
            "course_name": "Python Basics",
            "fit_score": 0.92,
            "expected_difficulty": 2,
            "reasons": ["Aligned with your learning style", "Good foundation for advanced topics"],
        },
        {
            "course_id": 2,
            "course_name": "Advanced Web Development",
            "fit_score": 0.88,
            "expected_difficulty": 4,
            "reasons": ["Strong recommendation based on GPA", "Matches target credit load"],
        },
        {
            "course_id": 3,
            "course_name": "Machine Learning Fundamentals",
            "fit_score": 0.85,
            "expected_difficulty": 4,
            "reasons": ["Perfect for your major", "Complements your prerequisite courses"],
        },
        {
            "course_id": 4,
            "course_name": "Database Design",
            "fit_score": 0.82,
            "expected_difficulty": 3,
            "reasons": ["Practical skills applicable to projects", "Good pace for your study hours"],
        },
        {
            "course_id": 5,
            "course_name": "Cloud Computing",
            "fit_score": 0.79,
            "expected_difficulty": 5,
            "reasons": ["Industry-relevant skills", "Matches your career goals"],
        },
    ]

    sorted_courses = sorted(
        mock_courses, key=lambda x: x["fit_score"], reverse=True
    )
    return sorted_courses[:top_k]
