from entity.request import HarReq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import tkhar

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


@api.get("/")
async def root():
    return {"message": "Hello World"}


@api.post("/har")
async def har(req: HarReq):
    print(tkhar.get_input(req.data).shape)
    return {"message": "Hello World"}
