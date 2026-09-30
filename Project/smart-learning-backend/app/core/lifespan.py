import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Dict toàn cục giữ model trong RAM – các service import trực tiếp biến này.
ml_models: dict[str, Any] = {}

# Thư mục gốc backend (chứa file .pkl)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: nạp model lên RAM khi startup,
    giải phóng khi shutdown → đảm bảo zero-delay cho mỗi request.
    """
    # ── Startup ──────────────────────────────────────────────
    logger.info("Đang nạp các model Machine Learning vào RAM...")

    # 1. Hybrid Teacher Recommender (Content-Based + SVD)
    #    SVD model được nạp bên trong class → không cần load riêng.
    try:
        from app.services.hybrid_recommender import HybridTeacherRecommender
        ml_models["teacher_hybrid"] = HybridTeacherRecommender(backend_dir=_BASE_DIR)
        logger.info("HybridTeacherRecommender initialized successfully.")
    except Exception:
        ml_models["teacher_hybrid"] = None
        logger.exception("Failed to initialize HybridTeacherRecommender – hybrid scores will be unavailable.")

    # 2. Pass/Fail Prediction (XGBoost + Ordinal Encoders)
    try:
        model_pkl = _BASE_DIR / "pass_prediction_model.pkl"
        enc_pkl = _BASE_DIR / "pass_prediction_encoders.pkl"
        if model_pkl.exists() and enc_pkl.exists():
            ml_models["pass_predictor"] = joblib.load(model_pkl)
            ml_models["pass_encoders"] = joblib.load(enc_pkl)
            logger.info("Loaded Pass/Fail model + encoders from %s", model_pkl)
        else:
            ml_models["pass_predictor"] = None
            ml_models["pass_encoders"] = None
            logger.warning("pass_prediction_model.pkl not found at %s", model_pkl)
    except Exception:
        ml_models["pass_predictor"] = None
        ml_models["pass_encoders"] = None
        logger.exception("Failed to load Pass/Fail model – predictions will be unavailable.")

    logger.info("Nạp model hoàn tất.")

    yield

    # ── Shutdown ─────────────────────────────────────────────
    ml_models.clear()
    logger.info("Đã giải phóng model khỏi bộ nhớ.")
