
# 🌳 AgarVision  
## AI-Powered Decision Support System for Agarwood Cultivation

### Integrated Quality Grading • Disease Detection • Induction Analysis • Market Intelligence

---

**AgarVision** is an AI-powered **Decision Support System for Agarwood Cultivation**, designed to transform traditional, manual agarwood practices into **data-driven, explainable, and scalable AI-assisted decision-making**.

The system is developed as a **Final Year Research Project** and consists of **four distinct but integrated research contributions**, each addressing a major limitation in the Sri Lankan agarwood industry.

Rather than being a single-model application, AgarVision functions as a **modular AI ecosystem**, supporting **cultivation, harvesting, quality assessment, export readiness, and market decision-making**.

---

## 📌 1. Project Overview

Agarwood cultivation and export decisions in Sri Lanka currently rely on:

- Manual inspection and expert judgment  
- Subjective quality grading  
- Late disease identification  
- Poor harvest timing decisions  
- Guesswork-based market and pricing strategies  

These limitations result in:

- Inconsistent export quality  
- Revenue loss due to poor timing and pricing  
- Reduced global competitiveness  

**AgarVision addresses these challenges through four AI-driven research modules**, integrated into a single mobile-based decision support platform.

---

## 🧠 2. Research Contributions (Modules)

### 1️⃣ Export Readiness & Resin Quality Assessment Module  
**Owner:** Resin Quality & Export Decision Component  

- Image-based quality grading of agarwood resin chips  
- Classifies resin into **Premium, Grade A, and Grade B**  
- Uses **CNN-based deep learning models**  
- Combines image-based quality output with numeric attributes  
- Generates:
  - Quality grade  
  - Export readiness result  
  - Reasons for rejection  
  - Improvement tips for export compliance  

**Impact:**  
Reduces subjectivity and improves consistency in export quality decisions.

---

### 2️⃣ Agarwood Leaf Disease Detection & Remedy Recommendation Module  
**Owner:** Disease Detection & Farmer Assistance Component  

- CNN-based image classification of agarwood leaf diseases  
- Detects **four common agarwood leaf diseases**  
- Displays disease information via mobile application  
- Provides **specific and actionable remedy suggestions**  
- Designed for real-time, field-level usage  

**Impact:**  
Enables early disease detection, preventing quality degradation and yield loss.

---

### 3️⃣ Agarwood Resin Induction Stage Classification Module  
**Owner:** Harvest Timing & Induction Analysis Component  

- Uses **numerical plantation data and bark images**  
- Applies **early multimodal data fusion**  
- Classifies trees as:
  - Too Early  
  - Ready  
  - Over-Mature  
- Eliminates dependency on expert inspection  
- Supports accurate harvest timing decisions  

**Impact:**  
Reduces risk of early or delayed harvesting using data-driven predictions.

---

### 4️⃣ Agarwood Oil Market Demand Forecasting & Price Recommendation Module  
**Owner:** Market Intelligence & Export Strategy Component  

- Analyzes historical export market data  
- Forecasts future demand using **Random Forest models**  
- Recommends suitable price ranges based on demand  
- Includes an **NLP-based chatbot** using an **SVM model**  
- Provides easy access to demand and pricing insights  

**Impact:**  
Helps exporters maximize revenue and reduce financial risk.

---

## ⭐ 3. Key Features

- Resin quality grading (Premium / Grade A / Grade B)  
- Export readiness evaluation with reasons and improvement tips  
- Agarwood leaf disease detection with remedy recommendations  
- Resin induction stage classification  
- Market demand forecasting and price recommendation  
- Chatbot-based user assistance  
- Explainable AI-driven outputs  
- Mobile-first, farmer-friendly UI  
- Modular and scalable architecture  

---

## 🏗️ 4. High-Level System Architecture

Users
(Farmers / Exporters / Plantation Staff)
        ↓
Mobile Application
(Expo + React Native)
        ↓
Backend API Layer
(Python Flask)
        ↓
AI & ML Models
 ├─ CNN Models (Image-based Analysis)
 ├─ Random Forest (Market Demand Forecasting)
 └─ SVM (Chatbot Intent Classification)
        ↓
Decision Outputs & Explanations
(Quality Grades • Readiness • Remedies • Market Insights)

<img width="2974" height="2224" alt="agarvision_high_level_architecture_visual" src="https://github.com/user-attachments/assets/117f460a-a333-4b05-9e6c-f81b9130261e" />

---

## 🔬 5. AI Pipeline Overview

### Image-Based AI (CNN)
- Resin image quality grading  
- Leaf disease detection  
- Bark image analysis for induction stage  

### Numerical & Tabular Models
- Random Forest for demand forecasting  
- Rule-based logic for price recommendation  

### NLP & Chatbot
- SVM-based intent classification  
- User-friendly natural language interaction  

---

## 📂 6. Project Structure

📱 Frontend (Expo + React Native)

agarvision/
├── app/                      
├── assets/                    
├── components/                
├── constants/                
├── contexts/                  
├── hooks/                     
├── lib/                       
├── scripts/                   
├── services/                
├── app.json
├── expo-env.d.ts
├── package.json
├── package-lock.json
├── tailwind.config.js
├── tsconfig.json
│
└── README.md

🧠 Backend (Python Flask – Modular Architecture)

agarvision-backend/
├── app/
│   ├── member_modules/              # Individual research components
│   │
│   │   ├── thisara_module/          # Disease Detection Module
│   │   │   ├── model/                
│   │   │   ├── predictor.py
│   │   │   ├── remedies.json
│   │   │   └── routes.py
│   │   │
│   │   ├── kavin_module/             # Resin Quality & Export Readiness
│   │   │   ├── model/
│   │   │   ├── predictor.py
│   │   │   ├── logic.py
│   │   │   └── routes.py
│   │   │
│   │   ├── oshini_module/            # Market Demand & Price Forecasting
│   │   │   ├── model/
│   │   │   ├── predictor.py
│   │   │   ├── chatbot.py
│   │   │   └── routes.py
│   │   │
│   │   └── thenuka_module/           # Resin Induction Stage Classifier
│   │       ├── model/
│   │       ├── predictor.py
│   │       └── routes.py
│   │
│   ├── __init__.py
│   └── main.py                       # Flask app entry point
│
├── tests/
│   └── test_health.py
│
├── requirements.txt
└── README.md


---

## 🛠️ 7. Tech Stack

### Frontend
- Expo  
- React Native  
- TypeScript  
- Expo Router  
- Axios  

### Backend
- Python Flask  
- RESTful APIs

### Database
- MongoDB

### AI / ML
- CNN (TensorFlow / Keras)  
- Random Forest  
- Support Vector Machine (SVM)  
- Scikit-learn  

---

## ⚙️ 8. Environment Variables

### Backend (`.env`)

- MONGO_URI = 
- JWT_SECRET =
- DISEASE_MODEL_PATH =
- RESIN_QUALITY_MODEL_PATH =
- RESIN_INDUCTION_MODEL_PATH = 
- DEMAND_MODEL_PATH =
- SVM_MODEL_PATH =

### Frontend (`.env`)

- EXPO_PUBLIC_API_URL = http://localhost:8081


---

## ▶️ 9. Setup & Installation

### Backend
```bash
cd Agarvision-backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
### Frontend
```bash
cd Agarvision
npm install
npx expo start -c
```

## 💼 10. Commercialization & Sustainability

- Uniqueness: First AI-driven decision support system for agarwood cultivation in Sri Lanka
- Market: Global luxury agarwood and oil export markets
- Deployment: Mobile application with planned Google Play release
- Cost Recovery: Subscription-based usage model
- Sustainability: Scalable architecture for long-term adoption

---

## 👥 11. Team 25-26J-266 (Specialization - IT)
- IT22315564 - Jayawardena L.P.G.K (Group Leader)
- IT22221346 - Kandage T.P
- IT22315328 - Rathnamalala R.M.B.I.T
- IT22267368 - Malmali W.S.V.M.O
