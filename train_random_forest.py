import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Veriyi yükle
csv_path = 'landmine_tabular_data.csv'
print("Veri yükleniyor...")
df = pd.read_csv(csv_path)

# Kullanılacak 5 özellik
features = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density']
target = 'label'

print(f"Toplam veri sayısı: {len(df)}")

# Eğitim ve test setlerini ayır
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

if len(train_df) == 0 or len(test_df) == 0:
    print("'split' sütunu bulunamadı, manuel olarak ayrılıyor...")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

print(f"Eğitim seti boyutu: {len(X_train)}")
print(f"Test seti boyutu: {len(X_test)}")

# Modeli oluştur ve eğit
print("\nRandom Forest modeli eğitiliyor...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Sonuçları değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Başarısı (Accuracy): {accuracy:.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Modeli kaydet
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel '{model_filename}' olarak kaydedildi.")

# İlk 5 veri için tahmin ve olasılıklar
print("\nTest setinden ilk 5 veri için mayın olma olasılıkları:")
probabilities = model.predict_proba(X_test.head(5))
predictions = model.predict(X_test.head(5))

for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
    print(f"Örnek {i+1}: Mayın Olma Olasılığı: %{prob[1]*100:.2f} -> Tahmin: {'Mayın (1)' if pred == 1 else 'Mayın Değil (0)'}")
