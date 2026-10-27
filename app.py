import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="TrustCheck ML",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "trustcheck_four_models.pkl"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

st.title("🛡️ TrustCheck ML")
st.subheader("Confidence-aware SMS spam classification")
st.write(
    "Classify an SMS as spam or ham. If the model's confidence is below "
    "the selected model's validation threshold, TrustCheck recommends human review."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Upload trustcheck_four_models.pkl to the same GitHub repository as app.py.")
    st.stop()

try:
    bundle = load_bundle()
    models = bundle["models"]
    thresholds = bundle["selected_thresholds"]
except Exception as exc:
    st.error(f"Could not load the model file: {exc}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Choose classification model", list(models.keys()))
    threshold = float(thresholds[model_name])
    st.metric("Validation-selected threshold", f"{threshold:.2f}")
    st.caption("Thresholds were selected using validation data. They do not guarantee correctness.")

st.markdown("---")
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### Enter an SMS message")
    message = st.text_area(
        "SMS message",
        placeholder="Example: Congratulations! You have won a free prize. Call now!",
        height=180,
        label_visibility="collapsed"
    )
    classify = st.button("Check message", type="primary", use_container_width=True)

    st.markdown("**Try a sample**")
    sample_choice = st.selectbox(
        "Sample messages",
        [
            "Choose a sample...",
            "Spam: Congratulations! You have won a free prize. Call now to claim your reward!",
            "Ham: Hey, are we meeting at college tomorrow? Please bring the notes.",
            "Ham: Hi, are you free to talk later today?"
        ]
    )
    if sample_choice != "Choose a sample...":
        st.info(sample_choice.split(": ", 1)[1])

with right:
    st.markdown("### TrustCheck result")
    if classify:
        if not message.strip():
            st.warning("Please enter an SMS message first.")
        else:
            model = models[model_name]
            probabilities = model.predict_proba([message])[0]
            ham_probability = float(probabilities[0])
            spam_probability = float(probabilities[1])
            predicted_class = "SPAM" if spam_probability >= 0.5 else "HAM"
            confidence = max(ham_probability, spam_probability)
            accepted = confidence >= threshold

            c1, c2 = st.columns(2)
            c1.metric("Predicted class", predicted_class)
            c2.metric("Model confidence", f"{confidence:.1%}")

            st.write("**Class probabilities**")
            st.progress(ham_probability, text=f"Ham: {ham_probability:.1%}")
            st.progress(spam_probability, text=f"Spam: {spam_probability:.1%}")

            if accepted:
                st.success(f"Automatic decision: {predicted_class}")
                st.caption("The model's confidence met the selected threshold.")
            else:
                st.warning("Human Review Required")
                st.caption("The confidence was below the selected threshold. This demo recommends review; it does not create a real reviewer queue.")

            with st.expander("Technical details"):
                st.write(f"Selected model: {model_name}")
                st.write(f"Decision threshold: {threshold:.2f}")
                st.write(f"Coverage policy: accept when confidence ≥ {threshold:.2f}")
    else:
        st.info("Enter a message and click **Check message** to see the prediction.")
        st.write("The app compares the model's confidence with its selected threshold.")

st.markdown("---")
st.caption(
    "Educational demonstration using the UCI SMS Spam Collection. "
    "Predictions can be wrong; do not treat model confidence as a guarantee."
)
