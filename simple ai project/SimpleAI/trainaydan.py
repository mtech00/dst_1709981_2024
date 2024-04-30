import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Veri setini yükle
data = pd.read_csv(r"./datasets/winequality-white.csv", sep=";" , on_bad_lines='skip')
data.fillna(0, inplace=True)

# Girdi (X) ve çıktı (y) değişkenlerini belirle
X = data.iloc[:, :11]  # İlk 11 sütunu girdi olarak kullan
y = data.iloc[:, 11]   # Son sütunu çıktı olarak kullan

# Veri setini eğitim ve test veri setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı Regresyon modelini eğit
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

# Eğitilmiş modeli dosyaya kaydet
with open(r'./white-wine_quality_predictor', 'wb') as predictor_Pickle:
    pickle.dump(dt_regressor, predictor_Pickle)

# Eğitim verileri üzerinde modelin doğruluğunu değerlendir
train_score = dt_regressor.score(X_train, y_train)
print("Coefficient of Correlation on Training Data:", train_score)
# R² skoru hesapla
train_r2_score = r2_score(y_train, dt_regressor.predict(X_train))
print("R² Score on Training Data:", train_r2_score)

# Test verileri üzerinde modelin doğruluğunu değerlendir
test_score = dt_regressor.score(X_test, y_test)
print("Coefficient of Correlation on Test Data:", test_score)
# R² skoru hesapla
test_r2_score = r2_score(y_test, dt_regressor.predict(X_test))
print("R² Score on Test Data:", test_r2_score)

# Eğitilmiş modeli kullanarak tahminler yap
predictions_train = dt_regressor.predict(X_train)
predictions_test = dt_regressor.predict(X_test)

# Eğitim veri seti üzerindeki tahminleri ekrana yazdır
print("Predictions on Training Data:", predictions_train)

# Test veri seti üzerindeki tahminleri ekrana yazdır
print("Predictions on Test Data:", predictions_test)