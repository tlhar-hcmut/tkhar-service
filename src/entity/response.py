from typing import List

from pydantic import BaseModel


class Response(BaseModel):
    code: int
    message: str


class HarPredict(BaseModel):
    id: int
    action: str
    confidence: float


class HarResponse(Response):
    data: List[HarPredict]
