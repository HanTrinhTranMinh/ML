# ==========================================================
# 🏠 VIETNAM HOUSING PRICE PREDICTION - LINEAR REGRESSION
# ==========================================================
# Author: <your name>
# Dataset: /kaggle/input/vietnam-housing-dataset-2024/vietnam_housing_dataset.csv
# ==========================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
from model import MyLinearRegression
from sklearn.linear_model import LinearRegression


# ==========================================================
# 1️⃣ ĐỌC DỮ LIỆU
# ==========================================================
file_path = os.path.join(os.path.dirname(__file__), "vietnam_housing_dataset.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file dataset tại: {file_path}")

df = pd.read_csv(file_path)
print("✅ Đọc dữ liệu thành công:", df.shape)
print(df.head())

# Phân loại diện tích
area_bins = [0, 40, 70, 150, np.inf]
area_labels = ['Under 40m', '40-70m', '70-150m', 'Tren 150m']
df['Area'] = pd.cut(df['Area'], bins=area_bins, labels=area_labels)

# ==========================================================
# 2️⃣ XỬ LÝ GIÁ TRỊ THIẾU THEO CỤM 'Address'
# ==========================================================
fill_with_mode = lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
df['Frontage'] = df.groupby('Address')['Frontage'].transform(fill_with_mode)
df['Access Road'] = df.groupby('Address')['Access Road'].transform(fill_with_mode)

print("\nGiá trị thiếu còn lại:")
print(df[['Frontage', 'Access Road']].isnull().sum())

# ==========================================================
# 3️⃣ XỬ LÝ CÁC CỘT KHÁC
# ==========================================================
df['House direction'].fillna('N/A', inplace=True)
df['Balcony direction'].fillna('N/A', inplace=True)

df['Floors'] = df.groupby('Address')['Floors'].transform(fill_with_mode)
df['Floors'].fillna(1, inplace=True)

# ==========================================================
# 4️⃣ XỬ LÝ BEDROOMS / BATHROOMS THEO MEDIAN
# ==========================================================
fill_with_median = lambda x: x.fillna(x.median())

df['Bedrooms'] = df.groupby(['Address'])['Bedrooms'].transform(fill_with_median)
df['Bathrooms'] = df.groupby(['Address'])['Bathrooms'].transform(fill_with_median)
df['Bedrooms'].fillna(1, inplace=True)
df['Bathrooms'].fillna(1, inplace=True)

# ==========================================================
# 5️⃣ XỬ LÝ GIÁ TRỊ DANH MỤC
# ==========================================================
df['Legal status'].fillna('Sale contract', inplace=True)
df['Furniture state'].fillna('N/A', inplace=True)

# ==========================================================
# 6️⃣ CHỌN ĐẶC TRƯNG & TARGET
# ==========================================================
num_features = ['Frontage', 'Access Road']
cat_features = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']
feature_cols = num_features + cat_features

X = df[feature_cols]
y = df['Price']  # Đơn vị: Tỷ VNĐ

# ==========================================================
# 7️⃣ CHIA DỮ LIỆU TRAIN / TEST
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================================
# 8️⃣ XÂY DỰNG PIPELINE XỬ LÝ DỮ LIỆU
# ==========================================================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Fit preprocessor
preprocessor.fit(X_train)

# Biến đổi dữ liệu
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==========================================================
# 9️⃣ HUẤN LUYỆN MÔ HÌNH LINEAR REGRESSION
# ==========================================================
model = MyLinearRegression(fit_intercept=True)
model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test_processed, y_test)
mase_val = model.mase(X_test_processed, y_test)

print(f"\n🎯 Đánh giá mô hình Linear Regression (Tự xây dựng):")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MASE: {mase_val:.4f}")

# ==========================================================
# 💾 LƯU MÔ HÌNH
# ==========================================================
model.save("linear_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("💾 Đã lưu linear_model.pkl và preprocessor.pkl")

# ==========================================================
# 📊 VẼ BIỂU ĐỒ THỰC TẾ VS DỰ ĐOÁN
# ==========================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', edgecolor='k')

# Vẽ đường chéo y = x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect Prediction (y = x)")

plt.title("📈 Actual vs Predicted House Prices (Tỷ VNĐ)", fontsize=13, fontweight='bold')
plt.xlabel("Actual Price (Tỷ VNĐ)", fontsize=11)
plt.ylabel("Predicted Price (Tỷ VNĐ)", fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==========================================================
# ⚖️ SO SÁNH VỚI SKLEARN.LINEAR_REGRESSION
# ==========================================================
sk_model = LinearRegression(fit_intercept=True)
sk_model.fit(X_train_processed, y_train)

my_model = MyLinearRegression(fit_intercept=True)
my_model.fit(X_train_processed, y_train)

print("⚖️ So sánh hệ số:", np.allclose(sk_model.coef_, my_model.coef_, atol=1e-6))
print("⚖️ So sánh intercept:", np.isclose(sk_model.intercept_, my_model.intercept_, atol=1e-6))

y_pred_sklearn = sk_model.predict(X_test_processed)
y_pred_my = my_model.predict(X_test_processed)
print("⚖️ Sai số trung bình giữa 2 mô hình:", np.mean(np.abs(y_pred_sklearn - y_pred_my)))
