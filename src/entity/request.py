from typing import List

from pydantic import BaseModel


class HarPoint(BaseModel):
    x: float
    y: float
    z: float


class HarReq(BaseModel):
    data: List[List[HarPoint]]
