# ==========================================================
# 🏠 VIETNAM HOUSING PRICE PREDICTION APP (STREAMLIT)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model import MyLinearRegression

# ==========================================================
# 1️⃣ LOAD MODEL & PREPROCESSOR
# ==========================================================

# ⚙️ PHẢI gọi đầu tiên
st.set_page_config(page_title="🏠 Vietnam Housing Price Prediction", layout="centered")

@st.cache_resource
def load_artifacts():
    """Load preprocessor và model (tự xây dựng)"""
    preprocessor = joblib.load("preprocessor.pkl")
    model_data = joblib.load("linear_model.pkl")

    # Tạo đối tượng model
    model = MyLinearRegression(fit_intercept=model_data.get("fit_intercept", True))

    # ✅ Hỗ trợ cả model mới (coef_/intercept_) và cũ (weights/bias)
    model.coef_ = model_data.get("coef_", model_data.get("weights"))
    model.intercept_ = model_data.get("intercept_", model_data.get("bias"))

    return preprocessor, model


# Load model & preprocessor
preprocessor, model = load_artifacts()
st.success("✅ Đã tải mô hình và preprocessor thành công!")

# ==========================================================
# 2️⃣ CẤU HÌNH GIAO DIỆN
# ==========================================================
st.title("🏠 Vietnam Housing Price Prediction")
st.markdown("### 📊 Dự đoán giá nhà (Linear Regression tự xây dựng)")
st.markdown("---")

# ==========================================================
# 3️⃣ INPUTS — NHẬP DỮ LIỆU TỪ NGƯỜI DÙNG
# ==========================================================
st.subheader("🧱 Nhập thông tin căn nhà:")

col1, col2 = st.columns(2)
with col1:
    frontage = st.number_input("🏗️ Mặt tiền (m)", min_value=1.0, max_value=30.0, value=5.0)
    access_road = st.number_input("🚗 Đường trước nhà (m)", min_value=1.0, max_value=50.0, value=6.0)
with col2:
    house_dir = st.selectbox("🧭 Hướng nhà", ["N/A", "Đông", "Tây", "Nam", "Bắc",
                                              "Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc"])
    balcony_dir = st.selectbox("🌅 Hướng ban công", ["N/A", "Đông", "Tây", "Nam", "Bắc",
                                                    "Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc"])

col3, col4 = st.columns(2)
with col3:
    legal_status = st.selectbox("📜 Tình trạng pháp lý", ["Have certificate", "Sale contract"])
with col4:
    furniture = st.selectbox("🛋️ Nội thất", ["Full", "Basic", "N/A"])

# ==========================================================
# 4️⃣ DỰ ĐOÁN
# ==========================================================
if st.button("🚀 Dự đoán giá nhà"):
    # ✅ Tạo DataFrame đầu vào
    input_data = pd.DataFrame([{
        "Frontage": frontage,
        "Access Road": access_road,
        "House direction": house_dir,
        "Balcony direction": balcony_dir,
        "Legal status": legal_status,
        "Furniture state": furniture
    }])

    # ✅ Tiền xử lý
    X_processed = preprocessor.transform(input_data)

    # ✅ Dự đoán giá (đơn vị: Tỷ VNĐ)
    price_billion = model.predict(X_processed)[0]  # KHÔNG chia /1000

    # ✅ Hiển thị kết quả
    st.success(f"💰 **Giá dự đoán:** {price_billion:.3f} tỷ VNĐ")
    st.metric(label="Predicted Price (Tỷ VNĐ)", value=f"{price_billion:.3f}")

    # Hiển thị biểu đồ nhỏ minh họa
    st.markdown("---")
    st.caption("📈 Mô hình Linear Regression tự xây dựng — kết quả dựa trên dữ liệu huấn luyện (đơn vị: Tỷ VNĐ)")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("📘 Developed by <your name> — Vietnam Housing Price Regression App")
