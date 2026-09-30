import asyncio
import logging

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.core.lifespan import ml_models

logger = logging.getLogger(__name__)


async def predict_pass_fail(
    gpa_bucket: str,
    study_hours_bucket: str,
    failed_subjects_count: str,
    attendance_bucket: str,
) -> dict:
    """
    Dự đoán xác suất PASS/FAIL bằng XGBoost.

    Luồng xử lý:
    1. Lấy model + bảng ordinal encoders từ ml_models (đã nạp trong lifespan).
    2. Encode input từ text → số thứ tự (ordinal) dựa trên bảng mapping.
    3. Gọi model.predict_proba() trong thread pool (CPU-bound).
    4. Xác định risk_level và trả về dict chuẩn cho schema.
    """
    model = ml_models.get("pass_predictor")
    encoders = ml_models.get("pass_encoders")

    if model is None or encoders is None:
        raise HTTPException(
            status_code=500,
            detail="Pass/Fail model chưa sẵn sàng. Kiểm tra log khởi động server.",
        )

    # ── Encode input theo ordinal map (đọc từ encoders.pkl) ──
    try:
        encoded = {
            "gpa_bucket": encoders["gpa_bucket"][gpa_bucket],
            "study_hours_bucket": encoders["study_hours_bucket"][study_hours_bucket],
            "failed_subjects_count": encoders["failed_subjects_count"][failed_subjects_count],
            "attendance_bucket": encoders["attendance_bucket"][attendance_bucket],
        }
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Giá trị không hợp lệ: {exc}. Kiểm tra lại các bucket đầu vào.",
        )

    # ── Tạo DataFrame đúng thứ tự cột như lúc train ──
    feature_order = ["gpa_bucket", "study_hours_bucket", "failed_subjects_count", "attendance_bucket"]
    input_df = pd.DataFrame([encoded], columns=feature_order)

    # ── Inference (CPU-bound) → chạy trong thread pool ──
    def _inference():
        proba = model.predict_proba(input_df)[0]  # [prob_fail, prob_pass]
        pred = model.predict(input_df)[0]          # 0 hoặc 1
        return proba, int(pred)

    proba, pred = await asyncio.to_thread(_inference)

    prob_pass = float(proba[1])
    prob_fail = float(proba[0])
    prediction_result = "PASS" if pred == 1 else "FAIL"

    # ── Xác định mức rủi ro ──
    if prob_pass > 0.8:
        risk_level = "LOW"
    elif prob_pass >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "probability_pass": round(prob_pass, 4),
        "probability_fail": round(prob_fail, 4),
        "prediction_result": prediction_result,
        "risk_level": risk_level,
    }
