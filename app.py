import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Cache model and scaler for better performance
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("diabetes_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# Title
st.title("🩺 Diabetes Prediction System")

st.markdown("""
This application predicts whether a patient is likely to have diabetes
based on medical diagnostic parameters.
""")

st.divider()

# Sidebar
st.sidebar.header("About Project")

st.sidebar.info("""
Machine Learning Project using:

- TensorFlow
- Streamlit
- SMOTE
- Neural Networks
""")

# Input Section
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, step=1)
    
    glucose = st.number_input("Glucose", min_value=0)

    bp = st.number_input("Blood Pressure", min_value=0)

    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)

    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        format="%.3f"
    )

    age = st.number_input("Age", min_value=0, step=1)

# Predict Button
if st.button("Predict Diabetes"):

    data = np.array([[preg, glucose, bp, skin,
                      insulin, bmi, dpf, age]])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    probability = prediction[0][0] * 100

    st.subheader("Prediction Result")

    if prediction[0][0] > 0.5:
        st.error(f"Patient is likely Diabetic")
    else:
        st.success(f"Patient is likely Non-Diabetic")

    st.write(f"### Diabetes Probability: {probability:.2f}%")

    st.progress(float(probability / 100))

st.divider()

st.caption("Developed using TensorFlow and Streamlit")