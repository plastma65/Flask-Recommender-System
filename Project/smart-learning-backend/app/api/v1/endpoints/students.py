import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter

from app.schemas.response import APIResponse, ResponseMeta

router = APIRouter()


@router.get("/me/profile", response_model=APIResponse[dict])
async def get_student_profile() -> APIResponse[dict]:
    
    profile_data = {
        "student_id": "550e8400-e29b-41d4-a716-446655440000",
        "major": "Computer Science",
        "gpa_bucket": "3.5-4.0",
        "attendance_bucket": "high",
        "learning_pref": {
            "time": "morning",
            "style": "interactive",
            "mode": "hybrid",
        },
    }
    
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    
    return APIResponse(
        success=True,
        message="Student profile retrieved successfully",
        data=profile_data,
        meta=ResponseMeta(
            request_id=request_id,
            timestamp=timestamp,
        ),
    )
