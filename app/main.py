from fastapi import FastAPI
from app.routes import health
from member_modules.thisara_disease.routes import router as thisara_router

app = FastAPI(title="AgarVision Backend", version="1.0.0")

# Common routes
app.include_router(health.router, tags=["Health"])

# Member module routes
app.include_router(thisara_router, prefix="/thisara", tags=["Thisara Disease"])

@app.get("/")
def root():
    return {"status": "ok", "message": "AgarVision backend running"}
