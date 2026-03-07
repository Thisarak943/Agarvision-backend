from fastapi import FastAPI
from app.routes import health
from member_modules.thisara_disease.routes import router as thisara_router
#Kavin imports
from member_modules.kavin_module.routes.image_routes import router as kavin_image_router
from member_modules.kavin_module.routes.numeric_routes import router as kavin_numeric_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AgarVision Backend", version="1.0.0")

# Common routes
app.include_router(health.router, tags=["Health"])

# Member module routes
app.include_router(thisara_router, prefix="/thisara", tags=["Thisara Disease"])

#Kavin routes
app.include_router(kavin_image_router, prefix="/kavin/image", tags=["Kavin Image Model"])
app.include_router(kavin_numeric_router, prefix="/kavin/numeric", tags=["Kavin Numeric Model"])

@app.get("/")
def root():
    return {"status": "ok", "message": "AgarVision backend running"}


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    