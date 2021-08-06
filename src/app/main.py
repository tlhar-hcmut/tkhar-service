from entity.request import HarReq
from entity.response import HarResponse, Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from serve import tkhar

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://20.205.205.211:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/")
async def home() -> Response:
    return Response(code=0, message="TKHAR Backend")


@api.post("/har")
async def har(req: HarReq) -> HarResponse:
    return tkhar.predict(req.data)
