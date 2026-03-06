from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health
from member_modules.thisara_disease.routes import router as thisara_router
from member_modules.thenuka_stage.stage_routes import router as thenuka_stage_router

app = FastAPI(title="AgarVision Backend", version="1.0.0")



# Common routes
app.include_router(health.router, tags=["Health"])

# Member module routes
app.include_router(thisara_router, prefix="/thisara", tags=["Thisara Disease"])
app.include_router(thenuka_stage_router, prefix="/thenuka/stage", tags=["Thenuka - Stage"])

@app.get("/")
def root():
    return {"status": "ok", "message": "AgarVision backend running"}