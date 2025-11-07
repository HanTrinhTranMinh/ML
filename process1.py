# ==========================================================
# 🏠 VIETNAM HOUSING PRICE PREDICTION - LINEAR REGRESSION (v2)
# ==========================================================
# Author: <your name>
# Dataset: ./vietnam_housing_dataset.csv
# ==========================================================

import os
import re
import unicodedata
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import MyOLSLinearRegression, MyPolynomialRegression


# ==========================================================
# 🧹 Outlier utils
# ==========================================================
def remove_outliers_iqr(df, cols, k=1.5):
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        if out[col].dtype.kind not in "bifc":
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        up = q3 + k * iqr
        out = out[(out[col] >= lo) & (out[col] <= up)]
    return out


# ==========================================================
# ⚙️ Helper
# ==========================================================
def compute_mase(y_true, y_pred, y_train):
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_train) < 2:
        return np.nan
    denom = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if denom == 0:
        return np.nan
    mae_model = np.mean(np.abs(y_true - y_pred))
    return mae_model / denom


def get_all_feature_names(preprocessor, num_features, cat_features):
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_features)
    num_feature_names = np.array(num_features)
    return np.concatenate([num_feature_names, cat_feature_names])


# ==========================================================
# 1️⃣ ĐỌC DỮ LIỆU
# ==========================================================
here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
file_path = os.path.join(here, "vietnam_housing_dataset.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file dataset tại: {file_path}")

df = pd.read_csv(file_path)
print("✅ Đọc dữ liệu thành công:", df.shape)
print(df.head())

numeric_cols_for_outlier = ['Area', 'Frontage', 'Access Road',
                            'Bedrooms', 'Bathrooms', 'Floors', 'Price']

df = remove_outliers_iqr(df, numeric_cols_for_outlier, k=1.5)

# ==========================================================
# 2️⃣ TIỀN XỬ LÝ ĐỊA CHỈ: chuẩn hoá, mở rộng viết tắt, tách Quận & City
# ==========================================================
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def expand_abbreviation(addr):
    patterns = {
        r"\bq\b": "quan",
        r"\bh\b": "huyen",
        r"\bp\b": "phuong",
        r"\btx\b": "thi xa",
        r"\btt\b": "thi tran",
        r"\btp\b": "thanh pho",
        r"\bx\b": "xa",
        r"\bt\b": "thon",
        r"\bkdc\b": "khu dan cu"
    }
    for pat, repl in patterns.items():
        addr = re.sub(pat, repl, addr)
    return addr


cities = [
    "ha noi", "ho chi minh", "da nang", "hai phong", "can tho",
    "an giang", "ba ria vung tau", "bac giang", "bac kan", "bac lieu", "bac ninh",
    "ben tre", "binh dinh", "binh duong", "binh phuoc", "binh thuan", "ca mau",
    "cao bang", "dak lak", "dak nong", "dien bien", "dong nai", "dong thap",
    "gia lai", "ha giang", "ha nam", "ha tinh", "hai duong", "hau giang", "hoa binh",
    "hung yen", "khanh hoa", "kien giang", "kon tum", "lai chau", "lam dong", "lang son",
    "lao cai", "long an", "nam dinh", "nghe an", "ninh binh", "ninh thuan", "phu tho",
    "phu yen", "quang binh", "quang nam", "quang ngai", "quang ninh", "quang tri",
    "soc trang", "son la", "tay ninh", "thai binh", "thai nguyen", "thanh hoa",
    "thua thien hue", "tien giang", "tra vinh", "tuyen quang", "vinh long",
    "vinh phuc", "yen bai"
]


def extract_city(addr):
    for city in cities:
        if city in addr:
            return city.title()
    return "Khác"


def extract_district(addr):
    district_patterns = [
        r"(quan\s*\d+)",
        r"(quan\s+[a-z\s]+)",
        r"(huyen\s+[a-z\s]+)",
        r"(thi\s+xa\s+[a-z\s]+)",
        r"(thi\s+tran\s+[a-z\s]+)"
    ]
    for pat in district_patterns:
        match = re.search(pat, addr)
        if match:
            name = match.group(1).strip().title()
            return (name.replace("Quan ", "Quận ")
                        .replace("Huyen ", "Huyện ")
                        .replace("Thi Xa ", "Thị xã ")
                        .replace("Thi Tran ", "Thị trấn "))
    return "Khác"


# Áp dụng trích xuất
df["Address_clean"] = df["Address"].astype(str).apply(normalize_text).apply(expand_abbreviation)
df["City"] = df["Address_clean"].apply(extract_city).str.title()
df["District"] = df["Address_clean"].apply(extract_district)

print("📍 Ví dụ trích xuất:")
print(df[["Address", "District", "City"]].head(10))

# ==========================================================
# 3️⃣ PHÂN LOẠI DIỆN TÍCH
# ==========================================================
area_bins = [0, 40, 70, 150, np.inf]
area_labels = ['Under 40m', '40-70m', '70-150m', 'Tren 150m']
df['AreaGroup'] = pd.cut(df['Area'], bins=area_bins, labels=area_labels)

# ==========================================================
# 4️⃣ XỬ LÝ GIÁ TRỊ THIẾU
# ==========================================================
fill_with_mode = lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
df['Frontage'] = df.groupby('City')['Frontage'].transform(fill_with_mode)
df['Access Road'] = df.groupby('City')['Access Road'].transform(fill_with_mode)
df['Floors'] = df.groupby('City')['Floors'].transform(fill_with_mode)
df['Floors'].fillna(1, inplace=True)

fill_with_median = lambda x: x.fillna(x.median())
df['Bedrooms'] = df.groupby(['City'])['Bedrooms'].transform(fill_with_median)
df['Bathrooms'] = df.groupby(['City'])['Bathrooms'].transform(fill_with_median)
df['Bedrooms'].fillna(1, inplace=True)
df['Bathrooms'].fillna(1, inplace=True)

df['House direction'].fillna('N/A', inplace=True)
df['Balcony direction'].fillna('N/A', inplace=True)
df['Legal status'].fillna('Sale contract', inplace=True)
df['Furniture state'].fillna('N/A', inplace=True)

# ==========================================================
# 5️⃣ CHỌN FEATURE & TARGET
# ==========================================================
num_features = ['Frontage', 'Access Road', 'Bedrooms', 'Bathrooms', 'Floors']
cat_features = ['House direction', 'Balcony direction',
                'Legal status', 'Furniture state', 'AreaGroup',
                'District', 'City']  # ✅ thêm District

X = df[num_features + cat_features].copy()
y = df['Price'].copy()

# ==========================================================
# 6️⃣ CHIA DỮ LIỆU TRAIN/TEST
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================================
# 7️⃣ PIPELINE & MÔ HÌNH
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

preprocessor.fit(X_train)
X_train_p = preprocessor.transform(X_train)
X_test_p = preprocessor.transform(X_test)

# Huấn luyện
model = MyOLSLinearRegression(fit_intercept=True)
model.fit(X_train_p, y_train)
y_pred = model.predict(X_test_p)

# ==========================================================
# 8️⃣ ĐÁNH GIÁ
# ==========================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mase = compute_mase(y_test, y_pred, y_train)

print("\n🎯 Kết quả MyOLSLinearRegression:")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.4f} tỷ VND")
print(f"MSE:  {mse:.4f}")
print(f"MASE: {mase:.4f}")

# ==========================================================
# 💾 LƯU FILE
# ==========================================================
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump({"coef_": getattr(model, "coef_", None),
             "intercept_": getattr(model, "intercept_", 0.0),
             "fit_intercept": True}, "linear_model.pkl")
print("💾 Đã lưu preprocessor.pkl và linear_model.pkl")

# ==========================================================
# 📊 BIỂU ĐỒ KIỂM TRA
# ==========================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label="Perfect Prediction (y=x)")
plt.xlabel("Actual Price (Tỷ VNĐ)")
plt.ylabel("Predicted Price (Tỷ VNĐ)")
plt.title("📈 Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()

residuals = y_test.values - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.5, edgecolor='k')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("🔍 Residuals vs Predicted")
plt.tight_layout()
plt.show()
