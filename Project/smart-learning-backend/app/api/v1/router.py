from fastapi import APIRouter

from app.api.v1.endpoints import catalog, predictions, recommendations, students

api_v1_router = APIRouter()

api_v1_router.include_router(catalog.router, prefix="/catalog", tags=["Catalog"])
api_v1_router.include_router(students.router, prefix="/students", tags=["Students"])
api_v1_router.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])
api_v1_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
