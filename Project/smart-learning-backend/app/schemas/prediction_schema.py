from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PassPredictionRequest(BaseModel):
    """Request body cho API dự đoán Pass/Fail."""
    gpa_bucket: str = Field(
        ...,
        description="GPA hiện tại (bucket tiếng Việt)",
        examples=["Từ 2.5 đến 3.19"],
    )
    study_hours_bucket: str = Field(
        ...,
        description="Số giờ tự học mỗi tuần",
        examples=["Từ 6 đến10 giờ"],
    )
    failed_subjects_count: str = Field(
        ...,
        description="Số môn đã rớt",
        examples=["0", "1", "2", "3", "Từ 4 môn trở lên"],
    )
    attendance_bucket: str = Field(
        ...,
        description="Tỷ lệ điểm danh",
        examples=[">90%", "70–90%", "50–70%", "< 50%"],
    )


class PassPredictionResponseData(BaseModel):
    """Response body trả về kết quả dự đoán."""
    probability_pass: float = Field(..., ge=0.0, le=1.0, description="Xác suất PASS (0-1)")
    probability_fail: float = Field(..., ge=0.0, le=1.0, description="Xác suất FAIL (0-1)")
    prediction_result: Literal["PASS", "FAIL"] = Field(..., description="Kết quả dự đoán")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Mức độ rủi ro")
