import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="SVR House Price Predictor",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title-box {
    background: linear-gradient(135deg,#2563eb,#9333ea);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 20px;
    color: white;
    margin-top: 20px;
}

.metric-box {
    background-color: #334155;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.predict-box {
    background: linear-gradient(135deg,#059669,#10b981);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown("""
<div class="title-box">
    <h1>Support Vector Regression</h1>
    <p>California Housing Price Prediction using SVR</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["PRICE"] = housing.target

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = df.drop("PRICE", axis=1)

y = df["PRICE"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ---------------------------------------------------
# SVR MODEL
# ---------------------------------------------------

model = SVR(
    kernel='rbf',
    C=100,
    gamma='scale'
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

# ---------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:.3f}")

c2.metric("RMSE", f"{rmse:.3f}")

c3.metric("R2 Score", f"{r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots()

ax.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

ax.set_xlabel("Actual Price")

ax.set_ylabel("Predicted Price")

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Predict House Price")

MedInc = st.slider("Median Income", 0.0, 15.0, 5.0)

HouseAge = st.slider("House Age", 1.0, 60.0, 20.0)

AveRooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)

AveBedrms = st.slider("Average Bedrooms", 1.0, 10.0, 2.0)

Population = st.slider("Population", 1.0, 40000.0, 1000.0)

AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)

Latitude = st.slider("Latitude", 32.0, 42.0, 35.0)

Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

if st.button("Predict Price"):

    features = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    st.markdown(
        f'''
        <div class="predict-box">
            Predicted House Price <br><br>
            ${prediction * 100000:,.2f}
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)