import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Veri setini yükle
print("📊 Veri seti yükleniyor...")
df = pd.read_csv('../python_backend/gelisim_kayitlari_dataset.csv')

print(f"🔢 Toplam kayıt sayısı: {len(df)}")
print(f"📋 Toplam sütun sayısı: {len(df.columns)}")

print("\n🧐 TÜM SÜTUNLAR VE VERİ TİPLERİ:")
print("="*50)
for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
    null_count = df[col].isnull().sum()
    unique_count = df[col].nunique()
    print(f"{i:2d}. {col:20s} | {str(dtype):10s} | Null: {null_count:4d} | Unique: {unique_count:4d}")

print("\n📊 İLK 5 KAYIT:")
print("="*80)
print(df.head())

print("\n📈 NUMERİK SÜTUNLARIN İSTATİSTİKLERİ:")
print("="*80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerik sütunlar: {numeric_cols}")
print(df[numeric_cols].describe())

print("\n🏷️ KATEGORİK SÜTUNLARIN DEĞERLERİ:")
print("="*80)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Kategorik sütunlar: {categorical_cols}")

for col in categorical_cols:
    print(f"\n{col.upper()} değerleri:")
    print(df[col].value_counts())

# Hedef değişkeni bul
print("\n🎯 HEDEF DEĞİŞKEN ANALİZİ:")
print("="*50)
if 'gunlukArtis' in df.columns:
    target = 'gunlukArtis' 
    print(f"Hedef değişken: {target}")
    print(f"Min: {df[target].min():.3f}")
    print(f"Max: {df[target].max():.3f}")
    print(f"Ortalama: {df[target].mean():.3f}")
    print(f"Standart sapma: {df[target].std():.3f}")
elif 'kilo' in df.columns:
    target = 'kilo'
    print(f"Alternatif hedef: {target}")
    print(f"Min: {df[target].min():.3f}")
    print(f"Max: {df[target].max():.3f}")
    print(f"Ortalama: {df[target].mean():.3f}")
    print(f"Standart sapma: {df[target].std():.3f}")
else:
    print("❌ Hedef değişken bulunamadı!")
    print("Mevcut sütunlar:", df.columns.tolist())
    exit()

# TÜM özellikleri hazırla
print("\n🔧 TÜM ÖZELLİKLER HAZIRLANİYOR...")
print("="*50)

# Kopya oluştur
df_ml = df.copy()

# Kategorik değişkenleri encode et
label_encoders = {}
for col in categorical_cols:
    if col != target:  # Hedef değişken kategorik değilse
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le
        print(f"✅ {col} encoded: {len(le.classes_)} farklı değer")

# Eksik değerleri doldur
print("\n🔧 EKSİK DEĞERLER DOLDuruluyor...")
for col in df_ml.columns:
    if df_ml[col].isnull().sum() > 0:
        if df_ml[col].dtype in ['int64', 'float64']:
            df_ml[col] = df_ml[col].fillna(df_ml[col].median())
            print(f"✅ {col}: {df[col].isnull().sum()} eksik değer median ile dolduruldu")
        else:
            df_ml[col] = df_ml[col].fillna(df_ml[col].mode()[0])
            print(f"✅ {col}: {df[col].isnull().sum()} eksik değer mode ile dolduruldu")

# Feature'ları ve target'ı ayır  
print(f"\n🎯 HEDEF DEĞİŞKEN: {target}")
y = df_ml[target]
X = df_ml.drop(columns=[target])

print(f"📊 Feature sayısı: {X.shape[1]}")
print(f"📋 Feature isimleri: {X.columns.tolist()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n🔄 TRAIN-TEST SPLIT:")
print(f"Train: {X_train.shape[0]} kayıt")
print(f"Test: {X_test.shape[0]} kayıt")

# Random Forest modeli - TÜM FEATURES İLE
print("\n🌲 RANDOM FOREST MODELİ EĞİTİLİYOR...")
print("="*50)

rf_model = RandomForestRegressor(
    n_estimators=200,        # Daha fazla ağaç
    max_depth=15,           # Daha derin ağaçlar
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1              # Tüm CPU core'ları kullan
)

rf_model.fit(X_train, y_train)

# Tahmin ve performans
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n📊 MODEL PERFORMANSI:")
print("="*50)
print(f"🎯 Train R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"🎯 Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"📏 MAE: {mae:.4f}")
print(f"📐 RMSE: {rmse:.4f}")

# FEATURE IMPORTANCE ANALİZİ - TÜM ÖZELLİKLER
print(f"\n🔍 FEATURE IMPORTANCE - TÜM {len(X.columns)} ÖZELLİK:")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("🏆 ÖNEMLİLİK SIRALAMASI:")
for i, (idx, row) in enumerate(feature_importance.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:20s}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")

# En önemli 10 feature
top_features = feature_importance.head(10)
print(f"\n⭐ EN ÖNEMLİ 10 ÖZELLİK:")
print("="*40)
for i, (idx, row) in enumerate(top_features.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:15s}: {row['importance']*100:.1f}%")

# Önemsiz features (<%1)
low_importance = feature_importance[feature_importance['importance'] < 0.01]
if len(low_importance) > 0:
    print(f"\n⚠️  DÜŞÜK ÖNEMLİLİKTE FEATURES (<%1):")
    print("="*40)
    for idx, row in low_importance.iterrows():
        print(f"   {row['feature']:15s}: {row['importance']*100:.2f}%")

print(f"\n📋 ÖZET:")
print("="*50)
print(f"🎯 Model başarısı: {test_r2*100:.1f}%")
print(f"📊 Toplam feature: {len(X.columns)}")
print(f"⭐ Yüksek önemli (>5%): {len(feature_importance[feature_importance['importance'] > 0.05])}")
print(f"🔸 Orta önemli (1-5%): {len(feature_importance[(feature_importance['importance'] >= 0.01) & (feature_importance['importance'] <= 0.05)])}")
print(f"⚪ Düşük önemli (<1%): {len(feature_importance[feature_importance['importance'] < 0.01])}")

# Model kaydet
import joblib
print(f"\n💾 MODEL KAYDEDILIYOR...")
joblib.dump(rf_model, 'comprehensive_random_forest_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("✅ Model dosyaları kaydedildi:")
print("   - comprehensive_random_forest_model.pkl")
print("   - label_encoders.pkl") 
print("   - feature_columns.pkl")

print(f"\n🎉 ANALİZ TAMAMLANDI!")
print(f"🚀 {test_r2*100:.1f}% doğrulukla TÜM {len(X.columns)} özellik kullanılıyor!") 