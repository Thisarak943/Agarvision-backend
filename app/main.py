from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
load_dotenv()  # loads .env into environment variables

from app.routes import health
from member_modules.thisara_disease.routes import router as thisara_router
from member_modules.oshini_module.demand_module.routes import router as demand_router
from member_modules.oshini_module.chatbot_module.chatbot_routes import router as chatbot_router



BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where main.py is
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

app = FastAPI(title="AgarVision Backend", version="1.0.0")

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