from entity.request import HarReq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from serve import tkhar

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.post("/har")
async def har(req: HarReq):
    return tkhar.predict(req.data)
