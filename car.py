import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Car Price Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# BACKGROUND IMAGE + CSS
# =====================================================

st.markdown(
    """
    <style>

    .stApp{
        background-image:url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7");
        background-size:cover;
        background-position:center;
        background-attachment:fixed;
    }

    .main-title{
        text-align:center;
        color:white;
        font-size:50px;
        font-weight:bold;
        margin-bottom:0px;
    }

    .sub-title{
        text-align:center;
        color:white;
        font-size:20px;
        margin-bottom:30px;
    }

    .glass{
        background: rgba(255,255,255,0.18);
        backdrop-filter: blur(12px);
        padding:20px;
        border-radius:15px;
        color:white;
        border:1px solid rgba(255,255,255,0.2);
    }

    .prediction{
        background:rgba(0,128,0,0.85);
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        font-size:38px;
        font-weight:bold;
    }

    .stButton button{
        width:100%;
        background:#2563eb;
        color:white;
        font-size:20px;
        border-radius:10px;
        height:55px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TITLE
# =====================================================

st.markdown(
    "<div class='main-title'>🚗 Car Price Prediction Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Predict Used Car Prices Using Machine Learning</div>",
    unsafe_allow_html=True
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("carprice.csv")

df.replace("?", np.nan, inplace=True)

numeric_cols = [
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df.dropna(inplace=True)

# =====================================================
# ENCODING
# =====================================================

fuel_encoder = LabelEncoder()
engine_location_encoder = LabelEncoder()
engine_type_encoder = LabelEncoder()

df["fuel-type"] = fuel_encoder.fit_transform(
    df["fuel-type"]
)

df["engine-location"] = engine_location_encoder.fit_transform(
    df["engine-location"]
)

df["engine-type"] = engine_type_encoder.fit_transform(
    df["engine-type"]
)

# =====================================================
# MODEL TRAINING
# =====================================================

X = df.drop("price", axis=1)

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🚘 Enter Car Details")

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    fuel_encoder.classes_
)

engine_location = st.sidebar.selectbox(
    "Engine Location",
    engine_location_encoder.classes_
)

engine_type = st.sidebar.selectbox(
    "Engine Type",
    engine_type_encoder.classes_
)

horsepower = st.sidebar.slider(
    "Horsepower",
    40,
    300,
    120
)

peak_rpm = st.sidebar.slider(
    "Peak RPM",
    4000,
    7000,
    5000
)

city_mpg = st.sidebar.slider(
    "City MPG",
    10,
    60,
    25
)

highway_mpg = st.sidebar.slider(
    "Highway MPG",
    10,
    70,
    35
)

predict = st.sidebar.button("🚀 Predict Price")

# =====================================================
# INFORMATION CARDS
# =====================================================

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='glass'>
    <h3>⛽ Fuel Type</h3>
    Different fuel types influence vehicle value and operating costs.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='glass'>
    <h3>⚙ Horsepower</h3>
    Higher horsepower generally increases vehicle performance and price.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='glass'>
    <h3>🛣 Mileage</h3>
    Better fuel efficiency often attracts more buyers.
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================
# PREDICTION
# =====================================================

if predict:

    fuel_value = fuel_encoder.transform(
        [fuel_type]
    )[0]

    location_value = engine_location_encoder.transform(
        [engine_location]
    )[0]

    engine_value = engine_type_encoder.transform(
        [engine_type]
    )[0]

    sample = np.array([[
        fuel_value,
        location_value,
        engine_value,
        horsepower,
        peak_rpm,
        city_mpg,
        highway_mpg
    ]])

    prediction = model.predict(sample)[0]

    st.markdown(
        f"""
        <div class='prediction'>
        💰 Predicted Car Price <br>
        ₹ {prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.subheader("📊 Selected Car Specifications")

    chart_data = pd.DataFrame({
        "Feature":[
            "Horsepower",
            "Peak RPM",
            "City MPG",
            "Highway MPG"
        ],
        "Value":[
            horsepower,
            peak_rpm,
            city_mpg,
            highway_mpg
        ]
    })

    st.bar_chart(
        chart_data.set_index("Feature")
    )

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.subheader("📈 Feature Importance")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.set_title(
    "Factors Affecting Car Price"
)

st.pyplot(fig)

# =====================================================
# DATA PREVIEW
# =====================================================

st.subheader("📄 Dataset Preview")

st.dataframe(df.head())