from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prediction import Prediction


async def create_pass_prediction(session: AsyncSession, data_in: dict) -> Prediction:
    db_prediction = Prediction(**data_in)
    session.add(db_prediction)
    await session.commit()
    await session.refresh(db_prediction)
    return db_prediction


async def get_prediction_by_student_and_course(
    session: AsyncSession, student_id: str, course_id: int
) -> Prediction | None:
    stmt = select(Prediction).where(
        (Prediction.student_id == student_id) & (Prediction.course_id == course_id)
    ).order_by(Prediction.prediction_id.desc()).limit(1)
    
    result = await session.execute(stmt)
    return result.scalars().first()
