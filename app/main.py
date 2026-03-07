from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
load_dotenv()  # loads .env into environment variables

from fastapi.middleware.cors import CORSMiddleware

from app.routes import health
from member_modules.thisara_disease.routes import router as thisara_router
from member_modules.oshini_module.demand_module.routes import router as demand_router
from member_modules.oshini_module.chatbot_module.chatbot_routes import router as chatbot_router



BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where main.py is
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
#Kavin imports
from member_modules.kavin_module.routes.image_routes import router as kavin_image_router
from member_modules.kavin_module.routes.numeric_routes import router as kavin_numeric_router
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Common routes
app.include_router(health.router, tags=["Health"])

# Member module routes
app.include_router(thisara_router, prefix="/thisara", tags=["Thisara Disease"])

#Kavin routes
app.include_router(kavin_image_router, prefix="/kavin/image", tags=["Kavin Image Model"])
app.include_router(kavin_numeric_router, prefix="/kavin/numeric", tags=["Kavin Numeric Model"])
app.include_router(thenuka_stage_router, prefix="/thenuka/stage", tags=["Thenuka - Stage"])
app.include_router(demand_router)
app.include_router(chatbot_router)

@app.get("/demo/chatbot", response_class=HTMLResponse)
def chatbot_demo(request: Request):
    return templates.TemplateResponse("chatbot_demo.html", {"request": request})

@app.get("/demo/demand", response_class=HTMLResponse)
def demand_demo(request: Request):
    return templates.TemplateResponse("demand_demo.html", {"request": request})

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