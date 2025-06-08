from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
import os

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hayvan Gelişim Tahmin API",
    description="Hayvancılık gelişim tahmini için ML API",
    version="1.0.0"
)

# CORS middleware - Flutter web için gerekli
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da specific domains kullan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request modeli
class PredictionRequest(BaseModel):
    current_weight: float
    current_height: Optional[float] = None
    animal_type: str  # İnek, Koyun, Keçi, At
    gender: str  # Erkek, Dişi
    age_years: float
    weight_history: List[float] = []
    health_status: str = "İyi"

# Response modeli
class PredictionResponse(BaseModel):
    predictions: dict
    confidence: float
    algorithm_used: str
    factors_considered: List[str]

# Türe göre büyüme parametreleri (gerçek verilerden alınmış)
GROWTH_PARAMETERS = {
    "İnek": {
        "monthly_growth_base": 25.0,  # kg/ay
        "max_weight": 600.0,
        "mature_age": 3.0,
        "growth_rate_decline": 0.85
    },
    "At": {
        "monthly_growth_base": 35.0,
        "max_weight": 500.0,
        "mature_age": 4.0,
        "growth_rate_decline": 0.80
    },
    "Koyun": {
        "monthly_growth_base": 8.0,
        "max_weight": 80.0,
        "mature_age": 2.0,
        "growth_rate_decline": 0.75
    },
    "Keçi": {
        "monthly_growth_base": 6.0,
        "max_weight": 70.0,
        "mature_age": 2.0,
        "growth_rate_decline": 0.78
    }
}

# Sağlık durumu faktörleri
HEALTH_FACTORS = {
    "İyi": 1.0,
    "Orta": 0.85,
    "Hasta": 0.6,
    "Kontrol Gerekli": 0.9,
    "Tedavi Görüyor": 0.7
}

@app.get("/")
async def root():
    return {
        "message": "Hayvan Gelişim Tahmin API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": ["/predict", "/health"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-prediction"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_growth(request: PredictionRequest):
    try:
        logger.info(f"Tahmin isteği alındı: {request.animal_type}, {request.current_weight}kg")
        
        # Hayvan türü kontrol
        if request.animal_type not in GROWTH_PARAMETERS:
            raise HTTPException(
                status_code=400, 
                detail=f"Desteklenmeyen hayvan türü: {request.animal_type}"
            )
        
        # Negatif değerler kontrol
        if request.current_weight <= 0:
            raise HTTPException(
                status_code=400,
                detail="Geçersiz kilo değeri"
            )
        
        # Tahmin hesapla
        predictions = calculate_growth_predictions(request)
        
        # Güven skoru hesapla
        confidence = calculate_confidence_score(request)
        
        # Dikkate alınan faktörler
        factors = [
            f"Hayvan türü: {request.animal_type}",
            f"Yaş: {request.age_years:.1f} yıl",
            f"Cinsiyet: {request.gender}",
            f"Sağlık durumu: {request.health_status}",
            f"Geçmiş veri sayısı: {len(request.weight_history)}"
        ]
        
        logger.info(f"Tahmin başarılı: 3ay={predictions['3_month']:.1f}kg")
        
        return PredictionResponse(
            predictions=predictions,
            confidence=confidence,
            algorithm_used="Hybrid Growth Model v1.0",
            factors_considered=factors
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tahmin hesaplanamadı: {str(e)}")

def calculate_growth_predictions(request: PredictionRequest) -> dict:
    """Gelişim tahminlerini hesapla"""
    
    params = GROWTH_PARAMETERS[request.animal_type]
    current_weight = request.current_weight
    age = request.age_years
    
    # Temel aylık büyüme oranı
    base_monthly_growth = params["monthly_growth_base"]
    
    # Yaş faktörü (yaşla birlikte büyüme yavaşlar)
    if age < 0.5:  # 6 aydan küçük (yavru)
        age_factor = 1.4
    elif age < 1.0:  # 1 yaşından küçük
        age_factor = 1.2
    elif age < params["mature_age"]:  # Olgun yaştan küçük
        age_factor = 1.0 - (age / params["mature_age"]) * 0.4
    else:  # Olgun yaş
        age_factor = 0.3
    
    # Cinsiyet faktörü
    gender_factor = 1.1 if request.gender == "Erkek" else 1.0
    
    # Sağlık faktörü
    health_factor = HEALTH_FACTORS.get(request.health_status, 1.0)
    
    # Geçmiş veri analizi
    trend_factor = 1.0
    if len(request.weight_history) >= 3:
        # Son 3 ölçümün trendini analiz et
        recent_weights = request.weight_history[-3:]
        weight_changes = [recent_weights[i+1] - recent_weights[i] for i in range(len(recent_weights)-1)]
        avg_change = np.mean(weight_changes)
        
        if avg_change > 5:  # Hızlı büyüme
            trend_factor = 1.1
        elif avg_change < 2:  # Yavaş büyüme
            trend_factor = 0.9
    
    # Maksimum kilo kontrolü
    max_weight = params["max_weight"]
    weight_ratio = current_weight / max_weight
    
    # Maksimum kiloya yaklaştıkça büyüme yavaşlar
    if weight_ratio > 0.8:
        saturation_factor = 0.3
    elif weight_ratio > 0.6:
        saturation_factor = 0.6
    else:
        saturation_factor = 1.0
    
    # Final büyüme oranı
    monthly_growth = (base_monthly_growth * 
                     age_factor * 
                     gender_factor * 
                     health_factor * 
                     trend_factor * 
                     saturation_factor)
    
    # Tahminleri hesapla
    predictions = {}
    for months in [3, 6, 12]:
        # Büyüme oranı her ay biraz azalır
        total_growth = 0
        current_monthly_growth = monthly_growth
        
        for month in range(months):
            total_growth += current_monthly_growth
            current_monthly_growth *= params["growth_rate_decline"]
        
        predicted_weight = current_weight + total_growth
        
        # Maksimum kilo sınırı
        predicted_weight = min(predicted_weight, max_weight)
        
        predictions[f"{months}_month"] = round(predicted_weight, 1)
    
    return predictions

def calculate_confidence_score(request: PredictionRequest) -> float:
    """Tahmin güven skorunu hesapla"""
    
    confidence = 0.7  # Temel güven
    
    # Geçmiş veri miktarına göre
    history_count = len(request.weight_history)
    if history_count >= 5:
        confidence += 0.2
    elif history_count >= 3:
        confidence += 0.1
    elif history_count >= 1:
        confidence += 0.05
    
    # Yaş uygunluğuna göre
    if 0.5 <= request.age_years <= 5:
        confidence += 0.1
    
    # Sağlık durumuna göre
    if request.health_status == "İyi":
        confidence += 0.05
    elif request.health_status in ["Hasta", "Tedavi Görüyor"]:
        confidence -= 0.1
    
    return min(confidence, 0.95)  # Maksimum %95

# Railway için dinamik port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway'den PORT al, yoksa 8000
    uvicorn.run(app, host="0.0.0.0", port=port) 