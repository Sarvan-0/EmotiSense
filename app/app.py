
import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(
    page_title="EmotiSense",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../models/emotisense_calibrated.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

EMOTION_ICONS = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲"
}

st.title("🧠 EmotiSense")
st.subheader("AI-powered Emotion Classification")

st.write(
    """
    EmotiSense uses Natural Language Processing and
    Machine Learning to identify emotions expressed
    in written text.
    """
)

st.divider()

st.subheader("Analyze your text")

user_text = st.text_area(
    "Enter your sentence",
    placeholder="Example: I feel incredibly happy today!",
    height=150,
    max_chars=2000
)

st.caption(f"Characters: {len(user_text)} / 2000")

if st.button("🔍 Analyze Emotion", type="primary"):

    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")

    else:
        with st.spinner("Analyzing your text..."):

            try:
                probabilities = model.predict_proba([user_text])[0]
                classes = model.classes_

                results = pd.DataFrame({
                    "Emotion": classes,
                    "Probability": probabilities
                })

                results = results.sort_values(
                    by="Probability",
                    ascending=False
                ).reset_index(drop=True)

                predicted_emotion = results.iloc[0]["Emotion"]
                confidence = results.iloc[0]["Probability"]

                icon = EMOTION_ICONS.get(predicted_emotion, "🧠")

                st.divider()
                st.subheader("Analysis Result")

                st.success(
                    f"{icon} Predicted Emotion: "
                    f"{predicted_emotion.upper()}"
                )

                st.metric(
                    label="Top predicted probability",
                    value=f"{confidence:.2%}"
                )

                st.subheader("Emotion Probability Distribution")

                chart_data = results.set_index("Emotion")

                st.bar_chart(
                    chart_data,
                    horizontal=True
                )

                st.subheader("Detailed Probabilities")

                display_results = results.copy()

                display_results["Emotion"] = (
                    display_results["Emotion"].map(
                        lambda x: (
                            f"{EMOTION_ICONS.get(x, '🧠')} "
                            f"{x.capitalize()}"
                        )
                    )
                )

                st.dataframe(
                    display_results.style.format({
                        "Probability": "{:.2%}"
                    }),
                    hide_index=True,
                    use_container_width=True
                )

                if confidence >= 0.80:
                    st.info(
                        "The model assigns a relatively high "
                        "probability to this prediction."
                    )
                elif confidence >= 0.60:
                    st.info(
                        "The model has moderate preference "
                        "for this predicted emotion."
                    )
                else:
                    st.warning(
                        "The model is uncertain. Multiple "
                        "emotions may be plausible."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.subheader("Try these examples")

examples = [
    "I am extremely happy today!",
    "I feel completely alone and miserable.",
    "I am terrified of what might happen.",
    "I am deeply in love with this person.",
    "I am furious about what happened.",
    "I cannot believe I won the lottery!"
]

for example in examples:
    st.code(example)

st.divider()

st.caption(
    """
    EmotiSense | Multi-class Emotion Classification

    Built using Python, Scikit-learn, TF-IDF and Streamlit.

    This is an experimental NLP model. Its predictions
    are not psychological assessments or diagnoses.
    """
)
