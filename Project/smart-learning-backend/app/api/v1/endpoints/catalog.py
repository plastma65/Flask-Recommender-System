import uuid
from pathlib import Path
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter

from app.schemas.response import APIResponse, ResponseMeta

router = APIRouter()


@router.get("/courses", response_model=APIResponse[list[dict]])
async def get_courses() -> APIResponse[list[dict]]:
    
    courses_data = [
        {"course_id": 1, "course_name": "Python Basics", "credit_count": 3},
        {"course_id": 2, "course_name": "Advanced Web Development", "credit_count": 4},
        {"course_id": 3, "course_name": "Machine Learning Fundamentals", "credit_count": 3},
        {"course_id": 4, "course_name": "Database Design", "credit_count": 3},
        {"course_id": 5, "course_name": "Cloud Computing", "credit_count": 4},
    ]
    
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    
    return APIResponse(
        success=True,
        message="Courses retrieved successfully",
        data=courses_data,
        meta=ResponseMeta(
            request_id=request_id,
            timestamp=timestamp,
        ),
    )


@router.get("/teachers", response_model=APIResponse[list[dict]])
async def get_teachers() -> APIResponse[list[dict]]:
    csv_path = Path(__file__).resolve().parents[5] / "Data" / "teacher_profiles_cleaned.csv"
    df = pd.read_csv(csv_path)
    teachers_data = df[["teacher_id", "teacher_name", "courses_taught"]].fillna("").to_dict("records")
    
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    
    return APIResponse(
        success=True,
        message="Teachers retrieved successfully",
        data=teachers_data,
        meta=ResponseMeta(
            request_id=request_id,
            timestamp=timestamp,
        ),
    )
