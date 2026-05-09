import streamlit as st
import joblib


model = joblib.load("sentiment_model.pkl")
bow_vectorizer = joblib.load("tfidf_vectorizer.pkl")


st.title("📊 Sentiment Analysis App")
st.write("Enter text below to analyze sentiment using Logistic Regression + Bag of Words.")


user_input = st.text_area("Enter your text:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":

        input_features = bow_vectorizer.transform([user_input])


        prediction = model.predict(input_features)[0]


        if prediction == 1:
            st.success("✅ Positive Sentiment")
        else:
            st.error("❌ Negative Sentiment")
    else:
        st.warning("Please enter some text before predicting.")
