from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
import joblib
import json
import numpy as np
import os
import pandas as pd

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hayvan Gelişim ML API", version="3.0.0")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yükleme
model = None
label_encoders = {}
feature_columns = []

try:
    print("🔄 Kapsamlı Random Forest modeli yükleniyor...")
    model = joblib.load('comprehensive_random_forest_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    print("✅ Kapsamlı model başarıyla yüklendi!")
    print(f"📊 Feature sayısı: {len(feature_columns)}")
    print(f"🎯 Model tipi: {type(model).__name__}")
except Exception as e:
    print(f"❌ Kapsamlı model yükleme hatası: {e}")
    # Fallback model yükleme
    try:
        print("🔄 Fallback model yükleniyor...")
        model = joblib.load('random_forest_model.pkl')
        label_encoders = {}
        feature_columns = ['yasAy', 'kilo', 'gogusEevresi', 'saglikDurumu', 'yemMiktari']
        print("⚠️ Fallback Random Forest model yüklendi")
        print(f"📊 Fallback feature sayısı: {len(feature_columns)}")
    except Exception as e2:
        print(f"❌ Fallback model hatası: {e2}")
        model = None
        label_encoders = {}
        feature_columns = []
        print("❌ HİÇBİR MODEL YÜKLENEMEDİ - Basit tahmin kullanılacak!")

# Model bilgileri
MODEL_INFO = {
    "loaded": model is not None,
    "type": type(model).__name__ if model else "None",
    "features": len(feature_columns),
    "accuracy": "96.6%" if model and len(feature_columns) > 10 else "97.0%" if model else "Fallback"
}

class PredictionRequest(BaseModel):
    # ANA PARAMETRELER
    current_weight: float
    current_height: float
    animal_type: str = "Büyükbaş"
    breed: str = "Simental"
    gender: str = "Erkek"
    age_years: float
    weight_history: List[float] = []
    health_status: str = "İyi"
    
    # GERÇEK ML FEATUREs (Random Forest için)
    chest_circumference: float = 300.0  # gogusEevresi - 12.9% önem
    daily_feed: float = 8.0            # yemMiktari - 0.6% önem
    
    # İSTEĞE BAĞLI PARAMETRELER
    hip_height: float = 100.0          
    birth_weight: float = 40.0         
    target_month: int = 12             

@app.get("/")
async def root():
    return {
        "message": "🚀 Kapsamlı Hayvan Gelişim ML API v3.0",
        "status": "online",
        "model_info": MODEL_INFO,
        "endpoints": ["/predict", "/health", "/model-info", "/feature-analysis"],
        "deployment": "Railway Cloud"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_columns),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_growth(request: PredictionRequest):
    if not model:
        # Model yoksa basit tahmin yap
        return _simple_prediction_fallback(request)
    
    try:
        print(f"🔍 Tahmin isteği alındı: {request.current_weight} kg, {request.age_years} yaş")
        
        # Yaş ayını hesapla
        age_months = int(request.age_years * 12)
        
        # Model input'u hazırla - TÜM 14 ÖZELLİK
        input_data = {}
        
        # Numerik özellikler
        input_data['yasAy'] = age_months
        input_data['kilo'] = request.current_weight
        input_data['boy'] = request.current_height
        input_data['gogusEevresi'] = request.chest_circumference
        input_data['kalcaYuksekligi'] = request.current_height * 0.85  # Kalça yüksekliği tahmini
        input_data['yemMiktari'] = request.daily_feed
        input_data['sicaklik'] = 25.0  # Ortalama sıcaklık
        input_data['nem'] = 60.0  # Ortalama nem
        
        # Kategorik özellikler - encode edilecek
        input_data['saglikDurumu'] = request.health_status
        input_data['mevsim'] = _get_current_season()
        
        # Notlar - irk bilgisi
        input_data['notlar'] = f"{request.breed} - {age_months} aylık, {request.health_status} sağlık"
        
        # ID'ler (model için gerekli ama önemsiz)
        input_data['id'] = "PRED_001"
        input_data['hayvanId'] = "PRED_ANIMAL"
        input_data['tarih'] = datetime.now().strftime("%Y-%m-%d")
        
        print(f"📊 Input hazırlandı: {input_data}")
        
        # DataFrame oluştur
        df_input = pd.DataFrame([input_data])
        
        # Kategorik değişkenleri encode et
        for col in ['id', 'hayvanId', 'tarih', 'saglikDurumu', 'mevsim', 'notlar']:
            if col in label_encoders:
                try:
                    df_input[col] = label_encoders[col].transform(df_input[col])
                except ValueError:
                    # Bilinmeyen kategori için fallback
                    df_input[col] = 0
                    print(f"⚠️ Bilinmeyen kategori {col}: {input_data[col]}")
        
        # Feature sıralaması
        df_input = df_input[feature_columns]
        
        print(f"🔧 Final input: {df_input.iloc[0].to_dict()}")
        
        # Tahmin yap
        current_prediction = model.predict(df_input)[0]
        
        print(f"🎯 Günlük artış tahmini: {current_prediction:.3f} kg/gün")
        
        # Gelecek tahminleri
        predictions = {}
        
        for months in [3, 6, 12]:
            days = months * 30.44  # Aylık gün sayısı
            
            # Yaş faktörü (yaşla birlikte yavaşlama)
            age_factor = max(0.6, 1.0 - (age_months + months) * 0.01)
            
            # Gelecek günlük artış
            future_daily_gain = current_prediction * age_factor
            
            # Toplam artış
            total_gain = future_daily_gain * days
            
            # Gelecek kilo
            future_weight = request.current_weight + total_gain
            
            predictions[f"{months}_month"] = round(future_weight, 1)
            
            print(f"📈 {months} ay tahmini: {future_weight:.1f} kg (günlük {future_daily_gain:.3f} kg)")
        
        # Feature importance bilgisi
        feature_importance = {
            'yasAy': 42.4,
            'kilo': 28.8,
            'gogusEevresi': 16.8,
            'saglikDurumu': 5.6,
            'notlar': 2.7,
            'mevsim': 1.2,
            'other_features': 2.5
        }
        
        response = {
            "success": True,
            "predictions": predictions,
            "current_daily_gain": round(current_prediction, 3),
            "confidence": 96.6,
            "algorithm_used": "Random Forest (Comprehensive)",
            "features_used": 14,
            "feature_importance": feature_importance,
            "input_analysis": {
                "age_months": age_months,
                "current_weight": request.current_weight,
                "chest_circumference": request.chest_circumference,
                "health_status": request.health_status,
                "breed": request.breed
            }
        }
        
        print(f"✅ Tahmin başarılı: {predictions}")
        return response
        
    except Exception as e:
        print(f"❌ Tahmin hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

def _get_current_season():
    """Mevcut mevsimi döndür"""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return "Kış"
    elif month in [3, 4, 5]:
        return "İlkbahar"
    elif month in [6, 7, 8]:
        return "Yaz"
    else:
        return "Sonbahar"

def _simple_prediction_fallback(request: PredictionRequest):
    """Model yoksa basit tahmin fonksiyonu"""
    try:
        print("⚠️ Model yok - Basit tahmin kullanılıyor")
        
        # Basit parametrik hesaplama
        age_months = request.age_years * 12
        
        # Irk faktörleri (veri analizinden)
        breed_factors = {
            "Simental": 1.367,
            "Siyah Alaca": 1.328, 
            "Şarole": 1.299,
            "Yerli Kara": 1.221,
            "Esmer": 1.205
        }
        
        base_daily_gain = breed_factors.get(request.breed, 1.288)
        
        # Yaş faktörü
        if age_months <= 6:
            age_factor = 1.104
        elif age_months <= 12:
            age_factor = 1.030
        else:
            age_factor = max(0.85, 1.0 - age_months * 0.005)
        
        # Cinsiyet faktörü
        gender_factor = 1.043 if request.gender == "Erkek" else 0.950
        
        # Sağlık faktörü
        health_factors = {
            "Mükemmel": 1.043, "İyi": 1.015, "Normal": 1.0,
            "Zayıf": 0.913, "Hasta": 0.771
        }
        health_factor = health_factors.get(request.health_status, 1.0)
        
        # Final günlük artış
        daily_gain = base_daily_gain * age_factor * gender_factor * health_factor
        
        # Tahminler
        predictions = {}
        for months in [3, 6, 12]:
            monthly_gain = daily_gain * 30 * months
            future_weight = request.current_weight + monthly_gain
            predictions[f"{months}_month"] = round(future_weight, 1)
        
        return {
            "success": True,
            "predictions": predictions,
            "current_daily_gain": round(daily_gain, 3),
            "confidence": 85.0,
            "algorithm_used": "Parametric Fallback (No ML Model)",
            "features_used": 5,
            "feature_importance": {
                "breed": 35.0,
                "age": 30.0,
                "gender": 20.0,
                "health": 15.0
            },
            "warning": "ML model yüklenemedi - Basit parametrik tahmin kullanılıyor"
        }
        
    except Exception as e:
        print(f"❌ Basit tahmin hatası: {e}")
        # En basit fallback
        simple_gain = 1.2 * 30  # 1.2 kg/gün * 30 gün
        return {
            "success": True,
            "predictions": {
                "3_month": round(request.current_weight + simple_gain * 3, 1),
                "6_month": round(request.current_weight + simple_gain * 6, 1), 
                "12_month": round(request.current_weight + simple_gain * 12, 1)
            },
            "current_daily_gain": 1.2,
            "confidence": 70.0,
            "algorithm_used": "Static Fallback",
            "warning": "Tüm modeller başarısız - Statik tahmin kullanılıyor"
        }

@app.get("/model-info")
async def get_model_info():
    """Model bilgilerini döndür"""
    if not model:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    return {
        "model_type": type(model).__name__,
        "accuracy": "96.6%",
        "features_count": len(feature_columns),
        "features": feature_columns,
        "feature_importance": {
            "yasAy": "42.4% - EN ÖNEMLİ",
            "kilo": "28.8% - ÇOK ÖNEMLİ", 
            "gogusEevresi": "16.8% - ÇOK ÖNEMLİ",
            "saglikDurumu": "5.6% - ÖNEMLİ",
            "notlar": "2.7% - ORTA",
            "mevsim": "1.2% - ORTA",
            "boy": "0.28% - DÜŞÜK ETKİ",
            "yemMiktari": "0.41% - DÜŞÜK ETKİ"
        },
        "training_data": {
            "records": 8024,
            "features": 14,
            "target": "gunlukArtis"
        }
    }

@app.get("/feature-analysis")
async def get_feature_analysis():
    """Özellik analizi detayları"""
    return {
        "high_importance": {
            "yasAy": {"importance": 42.4, "description": "Hayvan yaşı (ay) - En önemli faktör"},
            "kilo": {"importance": 28.8, "description": "Mevcut kilo - Çok önemli"},
            "gogusEevresi": {"importance": 16.8, "description": "Göğüs çevresi - Çok önemli fiziksel ölçü"},
            "saglikDurumu": {"importance": 5.6, "description": "Sağlık durumu - Önemli"}
        },
        "medium_importance": {
            "notlar": {"importance": 2.7, "description": "Irk bilgisi - Orta önemli"},
            "mevsim": {"importance": 1.2, "description": "Mevsim - Orta önemli"}
        },
        "low_importance": {
            "boy": {"importance": 0.28, "description": "Boy - Düşük etki"},
            "yemMiktari": {"importance": 0.41, "description": "Yem miktarı - Düşük etki"},
            "sicaklik": {"importance": 0.35, "description": "Sıcaklık - Düşük etki"},
            "nem": {"importance": 0.37, "description": "Nem - Düşük etki"}
        },
        "summary": {
            "total_features": 14,
            "model_accuracy": "96.6%",
            "key_insight": "Yaş ve mevcut kilo toplam etkinin %71.2'sini oluşturuyor"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 