from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recommendation import RecommendationResult


async def create_recommendation_batch(session: AsyncSession, results: list[dict]) -> None:
    if not results:
        return
    
    stmt = insert(RecommendationResult).values(results)
    await session.execute(stmt)
    await session.commit()


async def get_top_teacher_recommendations(
    session: AsyncSession, student_id: str, course_id: int, limit: int = 5
) -> list[RecommendationResult]:
    stmt = select(RecommendationResult).where(
        (RecommendationResult.student_id == student_id)
        & (RecommendationResult.course_id == course_id)
        & (RecommendationResult.target_type == "teacher")
    ).order_by(RecommendationResult.recommendation_rank.asc()).limit(limit)
    
    result = await session.execute(stmt)
    return list(result.scalars())
