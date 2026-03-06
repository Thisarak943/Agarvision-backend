from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health
from member_modules.thisara_disease.routes import router as thisara_router
from member_modules.thenuka_stage.stage_routes import router as thenuka_stage_router

app = FastAPI(title="AgarVision Backend", version="1.0.0")

#cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Common routes
app.include_router(health.router, tags=["Health"])

# Member module routes
app.include_router(thisara_router, prefix="/thisara", tags=["Thisara Disease"])
app.include_router(thenuka_stage_router, prefix="/thenuka/stage", tags=["Thenuka - Stage"])

@app.get("/")
def root():
    return {"status": "ok", "message": "AgarVision backend running"}