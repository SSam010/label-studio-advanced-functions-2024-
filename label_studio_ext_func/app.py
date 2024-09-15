import os
import sys

from pathlib import Path

import nltk

sys.path.append(os.path.join(sys.path[0], Path(__file__).parent.absolute()))

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from logger import config as logger
from api.router import router
from config import SWAGGER, TITLE

logger.setup_logger()

app = FastAPI(
    title=TITLE,
    docs_url="/docs" if SWAGGER else None,
    redoc_url="/redoc" if SWAGGER else None,
)


@app.on_event("startup")
async def startup_event():
    nltk.download("punkt", quiet=True)


app.include_router(router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/front", StaticFiles(directory="front"), name="front")


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    return RedirectResponse(url="/front/image/favicon.ico")
