# ==========================================================
# 🏠 VIETNAM HOUSING PRICE PREDICTION APP (STREAMLIT v2)
# ==========================================================
# Model: MyOLSLinearRegression (R² ≈ 0.48, MAE ≈ 1.20 tỷ)
# Dataset: vietnam_housing_dataset.csv
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model import MyOLSLinearRegression

# ==========================================================
# 1️⃣ CẤU HÌNH APP & LOAD MÔ HÌNH
# ==========================================================
st.set_page_config(page_title="🏠 Vietnam Housing Price Prediction", layout="centered")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    model_data = joblib.load("linear_model.pkl")

    model = MyOLSLinearRegression(fit_intercept=model_data.get("fit_intercept", True))
    model.coef_ = model_data.get("coef_", model_data.get("weights"))
    model.intercept_ = model_data.get("intercept_", model_data.get("bias"))
    return preprocessor, model

preprocessor, model = load_artifacts()
st.success("✅ Đã tải mô hình & preprocessor (R² ≈ 0.48, MAE ≈ 1.20 tỷ VNĐ)")

# ==========================================================
# 2️⃣ GIAO DIỆN NHẬP DỮ LIỆU
# ==========================================================
st.title("🏠 Vietnam Housing Price Prediction")
st.markdown("#### 📊 Dự đoán giá nhà theo mô hình Linear Regression (tự xây dựng)")
st.caption("Phiên bản mô hình: MyOLSLinearRegression (R² ≈ 0.48, MAE ≈ 1.20 tỷ VNĐ)")
st.markdown("---")

st.subheader("🧱 Nhập thông tin căn nhà:")

# --- Nhập liệu ---
col1, col2, col3 = st.columns(3)
with col1:
    frontage = st.number_input("🏗️ Mặt tiền (m)", min_value=1.0, max_value=50.0, value=5.0)
    access_road = st.number_input("🚗 Đường trước nhà (m)", min_value=1.0, max_value=50.0, value=6.0)
with col2:
    bedrooms = st.number_input("🛏️ Số phòng ngủ", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("🚿 Số phòng tắm", min_value=1, max_value=10, value=2)
with col3:
    floors = st.number_input("🏢 Số tầng", min_value=1, max_value=10, value=2)
    area_group = st.selectbox("📐 Nhóm diện tích", ['Under 40m', '40-70m', '70-150m', 'Tren 150m'])

col4, col5 = st.columns(2)
with col4:
    house_dir = st.selectbox("🧭 Hướng nhà", ["N/A", "Đông", "Tây", "Nam", "Bắc",
                                              "Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc"])
    balcony_dir = st.selectbox("🌅 Hướng ban công", ["N/A", "Đông", "Tây", "Nam", "Bắc",
                                                    "Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc"])
with col5:
    legal_status = st.selectbox("📜 Pháp lý", ["Have certificate", "Sale contract"])
    furniture = st.selectbox("🛋️ Nội thất", ["Full", "Basic", "N/A"])

# --- Thêm District + City ---
st.markdown("### 🗺️ Khu vực địa lý")
city = st.selectbox("🏙️ Thành phố", [
    "Ho Chi Minh", "Ha Noi", "Da Nang", "Hai Phong", "Can Tho",
    "Binh Duong", "Dong Nai", "Khanh Hoa", "Long An", "Quang Ninh",
    "Thua Thien Hue", "Bac Ninh", "Nghe An", "Khác"
])

# Gợi ý quận phổ biến theo City
district_options = {
    "Ho Chi Minh": ["Quận 1", "Quận 3", "Quận 5", "Quận 7", "Quận 10", "Bình Thạnh", "Tân Bình", "Gò Vấp", "Khác"],
    "Ha Noi": ["Ba Đình", "Hoàn Kiếm", "Đống Đa", "Cầu Giấy", "Thanh Xuân", "Long Biên", "Nam Từ Liêm", "Khác"],
    "Da Nang": ["Hải Châu", "Thanh Khê", "Ngũ Hành Sơn", "Sơn Trà", "Liên Chiểu", "Khác"],
    "Hai Phong": ["Ngô Quyền", "Lê Chân", "Hồng Bàng", "An Dương", "Khác"],
    "Can Tho": ["Ninh Kiều", "Bình Thủy", "Cái Răng", "Khác"]
}
district_list = district_options.get(city, ["Khác"])
district = st.selectbox("📍 Quận / Huyện", district_list)

# ==========================================================
# 3️⃣ DỰ ĐOÁN
# ==========================================================
if st.button("🚀 Dự đoán giá nhà", use_container_width=True):
    input_df = pd.DataFrame([{
        "Frontage": frontage,
        "Access Road": access_road,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "House direction": house_dir,
        "Balcony direction": balcony_dir,
        "Legal status": legal_status,
        "Furniture state": furniture,
        "AreaGroup": area_group,
        "District": district,
        "City": city
    }])

    # Tiền xử lý & dự đoán
    X_processed = preprocessor.transform(input_df)
    y_pred = model.predict(X_processed)[0]
    y_pred = max(y_pred, 0)

    # ======================================================
    # HIỂN THỊ KẾT QUẢ
    # ======================================================
    st.success(f"💰 **Giá dự đoán:** {y_pred:.2f} tỷ VNĐ")
    st.metric(label="Predicted Price", value=f"{y_pred:.2f} Tỷ VNĐ")

    st.markdown("---")
    st.caption("""
    📘 Mô hình Linear Regression (R² ≈ 0.48, MAE ≈ 1.20 tỷ VNĐ)  
    Dự đoán dựa trên các biến: Mặt tiền, Đường, Số phòng, Pháp lý, Hướng, Quận, Thành phố, ...
    """)

    chart_df = pd.DataFrame({
        "Thấp hơn thị trường (ước)": [y_pred * 0.9],
        "Giá dự đoán": [y_pred],
        "Cao hơn thị trường (ước)": [y_pred * 1.1]
    }).T.reset_index()
    chart_df.columns = ["Loại", "Giá (tỷ VNĐ)"]
    st.bar_chart(chart_df.set_index("Loại"))

# ==========================================================
# 4️⃣ FOOTER
# ==========================================================
st.markdown("---")
st.caption("📈 Developed by <your name> — Vietnam Housing Price Prediction (R² ≈ 0.48)")
