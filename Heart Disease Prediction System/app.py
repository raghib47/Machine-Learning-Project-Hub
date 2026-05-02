import streamlit as st
import pandas as pd
import joblib

model = joblib.load('SVM_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction")
st.markdown("Provide the following detail")

age = st.slider("Age", 18, 100,40)
sex = st.selectbox("Sex",['M','F'])
chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resiting_bp = st.number_input("Resting Blood Pressure (mm Hg)",80,200,120)
choleserol = st.number_input("Cholestrol Glucose (mm)",100,600,200)
fasting_bs = st.selectbox("Fsting Blood Sugar > 120 mg/dl",[0,1])
resting_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr = st.slider("Macx Heart Rate",60,220,150)
exercise_algina = st.selectbox("Exercise-Induced Algina",["Y","N"])
oldpeak = st.slider("Oldpeak  (ST Depression)",0.0,6.0,1.0)
st_slop = st.selectbox("ST Slope",["UP","Flat","Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "Resting Blood Pressure (mm Hg)": resiting_bp,
        "Cholestrol Glucose (mm)": choleserol,
        "Fasting Blood Sugar (mg/dl)": fasting_bs,
        "Max Heart Rate (mmHg)": max_hr,
        "Oldpeak (ST Depression)": st_slop,
        "Sex" + sex: 1,
        "Chest Pain Type": chest_pain,
        "Resting ECG": resting_ecg,
        "Exercise Algina": exercise_algina,
        "ST_Slope": st_slop,
    }

    input_df = pd.DataFrame([raw_input])
    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = 0
    input_df = input_df[expected_columns]
    Scaled_input = scaler.transform(input_df)
    prediction = model.predict(Scaled_input)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low risk of Heart Disease")



