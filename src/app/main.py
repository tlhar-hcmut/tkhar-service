from entity.request import HarReq
from entity.response import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from serve import tkhar

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:3000",
        "http://20.205.205.211",
        "http://20.205.205.211:80",
        "http://20.205.205.211:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api.mount("/static", StaticFiles(directory="static"), name="static")


@api.get("/")
async def home() -> Response:
    return Response(code=0, message="TKHAR Backend")


@api.post("/har")
async def har(req: HarReq) -> Response:
    return tkhar.predict(req.data)
