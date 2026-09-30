import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter

from app.schemas.response import APIResponse, ResponseMeta
from app.schemas.prediction_schema import (
    PassPredictionRequest,
    PassPredictionResponseData,
)
from app.services.ml_service import predict_pass_fail

router = APIRouter()


@router.post("/pass-fail", response_model=APIResponse[PassPredictionResponseData])
async def predict_pass_fail_endpoint(
    request: PassPredictionRequest,
) -> APIResponse[PassPredictionResponseData]:
    """
    Endpoint dự đoán kết quả học tập (Pass/Fail).
    Nhận features từ Frontend, gọi XGBoost model, trả về xác suất + kết quả.
    """

    # Gọi service layer để inference
    raw_result = await predict_pass_fail(
        gpa_bucket=request.gpa_bucket,
        study_hours_bucket=request.study_hours_bucket,
        failed_subjects_count=request.failed_subjects_count,
        attendance_bucket=request.attendance_bucket,
    )

    prediction_data = PassPredictionResponseData(**raw_result)

    request_id = str(uuid.uuid4())
    timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()

    return APIResponse(
        success=True,
        message="Dự đoán thành công",
        data=prediction_data,
        meta=ResponseMeta(
            request_id=request_id,
            timestamp=timestamp,
        ),
    )
