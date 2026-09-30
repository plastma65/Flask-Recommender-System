from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

# Type alias giúp rút gọn khai báo dependency trong endpoint.
DBSession = Annotated[AsyncSession, Depends(get_db)]

# Placeholder: trích student_id từ JWT (sẽ do team Auth triển khai).
MOCK_STUDENT_ID = "550e8400-e29b-41d4-a716-446655440000"


async def get_current_student_id() -> str:
    """Trả về student_id hiện tại. Thay bằng JWT decode khi tích hợp Auth."""
    return MOCK_STUDENT_ID
