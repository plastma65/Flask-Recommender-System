from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ResponseMeta(BaseModel):
    request_id: str
    timestamp: str


class APIResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: T
    meta: Optional[ResponseMeta] = None
