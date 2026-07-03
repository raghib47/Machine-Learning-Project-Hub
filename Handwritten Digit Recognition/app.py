import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model("digit_model.keras")


st.set_page_config(page_title="Digit Recognition", page_icon="✍️", layout="centered")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) and click **Predict**.")


canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


if st.button("Predict"):

    if canvas_result.image_data is not None:


        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        ).convert("L")


        img = img.resize((28, 28))


        img = np.array(img)


        img = img / 255.0


        img = 1 - img


        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"### Predicted Digit: {digit}")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Prediction Probabilities")

        for i, prob in enumerate(prediction[0]):
            st.progress(float(prob))
            st.write(f"{i}: {prob*100:.2f}%")

        st.image(img.reshape(28, 28), width=150, caption="Processed Image")
