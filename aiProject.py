import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Veri setini yükleme
data = pd.read_excel('AllData.xlsx')

# Sütun isimlerini düzeltme (Wire Widhth -> Wire Width)
data.rename(columns={'Wire Widhth': 'Wire Width'}, inplace=True)

# NaN değerleri kontrol et ve temizle
print("NaN değerleri olan satırlar:")
print(data[data['Time'].isna()])

# NaN değerleri olan satırları çıkar
data = data.dropna(subset=['Time'])
print(f"\nVeri temizleme sonrası kalan satır sayısı: {len(data)}")

# Özellikler ve hedef değişken
X = data.drop('Time', axis=1)
y = data['Time']

# Veri setinde kalan NaN değerleri kontrol et
if X.isna().any().any():
    print("\nÖzelliklerde NaN değerleri var:")
    print(X.isna().sum())
    # Özellik sütunlarındaki NaN değerleri 0 ile doldur
    X = X.fillna(0)

# Eğitim ve test setleri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modeli
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test seti üzerinde tahmin
y_pred = model.predict(X_test)

# Performans metrikleri
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performansı:")
print(f"MAE: {mae:.2f}, R²: {r2:.2f}")

# Çeşitli süre hesaplamaları
mean_time = y.mean()  # Ortalama süre
optimal_time = y.quantile(0.25)  # En iyi %25'lik dilimin süresi
typical_time = y.median()  # Medyan (tipik) süre
max_efficient_time = y.quantile(0.75)  # Verimli maksimum süre

print("\nSüre Analizi:")
print(f"En İyi Süre (25. percentil): {optimal_time:.1f} saat")
print(f"Tipik Süre (medyan): {typical_time:.1f} saat")
print(f"Ortalama Süre: {mean_time:.1f} saat")
print(f"Verimli Maksimum Süre (75. percentil): {max_efficient_time:.1f} saat")

# Yeni transformer için tahmin
new_transformer = pd.DataFrame({
    'Number of Transformers': [6],
    'Overall Coil Number': [18],
    'Power': [300],
    'Number Of Turn': [42],
    'Cooling Duct': [2],
    'Winding Height': [329],
    'Number Of Layers': [4],
    'Wire Width': [25.45],
    'Wire Thickness': [3.35],
    'Wire Type': [1],
    'AG': [0]
})

# Sütun sırasının eğitim verisiyle aynı olduğundan emin olalım
new_transformer = new_transformer[X.columns]

predicted_time = model.predict(new_transformer)
print(f"\nÖrnek Tahmin:")
print(f"Tahmin Edilen Süre: {predicted_time[0]:.1f} saat")

# Modeli ve süre istatistiklerini kaydet
model_data = {
    'model': model,
    'feature_order': list(X.columns),
    'mean_time': float(mean_time),
    'optimal_time': float(optimal_time),
    'typical_time': float(typical_time),
    'max_efficient_time': float(max_efficient_time)
}

joblib.dump(model_data, 'ag_yassi_tel.joblib')



