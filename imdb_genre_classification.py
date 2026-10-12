import streamlit as st
import pickle
import numpy as np
import os
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="IMDb Genre Predictor",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Resources
# -----------------------------
@st.cache_resource
def load_resources():
    model_path = r"L:\Softwares\VS Code\Files\RNN\imdb_genre_classification\imdb_genre_simplernn.h5"
    tokenizer_path = r"L:\Softwares\VS Code\Files\RNN\imdb_genre_classification\tokenizer_imdb_genre.pickle"
    max_length_path = r"L:\Softwares\VS Code\Files\RNN\imdb_genre_classification\max_length_imdb_genre.pickle"
    mlb_path = r"L:\Softwares\VS Code\Files\RNN\imdb_genre_classification\mlb_imdb_genre.pickle"

    missing = []
    for path in [model_path, tokenizer_path, max_length_path, mlb_path]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        st.error("❌ Missing files:")
        for m in missing:
            st.write(f"- `{m}`")
        st.info("Please put all 4 files in the same folder as app.py")
        st.stop()

    # Load model (compile=False to avoid version issues)
    model = load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(max_length_path, "rb") as f:
        max_length = pickle.load(f)

    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    return model, tokenizer, max_length, mlb

model, tokenizer, max_length, mlb = load_resources()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Prediction Function
# -----------------------------
def predict_genres(text: str, threshold: float = 0.5):
    if not text.strip():
        return [], []

    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")

    pred_probs = model.predict(padded, verbose=0)[0]

    # Get genres above threshold
    predicted_indices = np.where(pred_probs >= threshold)[0]
    predicted_genres = [mlb.classes_[i] for i in predicted_indices]
    predicted_scores = [pred_probs[i] for i in predicted_indices]

    # Sort by probability (highest first)
    sorted_pairs = sorted(zip(predicted_genres, predicted_scores), key=lambda x: x[1], reverse=True)
    
    if sorted_pairs:
        genres, scores = zip(*sorted_pairs)
        return list(genres), list(scores)
    else:
        # If nothing above threshold, return top 3
        top_indices = np.argsort(pred_probs)[-3:][::-1]
        genres = [mlb.classes_[i] for i in top_indices]
        scores = [pred_probs[i] for i in top_indices]
        return genres, scores

# -----------------------------
# UI
# -----------------------------
st.title("🎭 IMDb Genre Predictor")
st.markdown("Paste a movie/TV show **plot description** and the model will predict its **genres**.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.2,
        max_value=0.8,
        value=0.45,
        step=0.05,
        help="Higher threshold = fewer but more confident genres"
    )
    
    st.markdown("---")
    st.header("Model Info")
    st.write("- Model: **SimpleRNN**")
    st.write(f"- Max sequence length: `{max_length}`")
    st.write(f"- Number of genres: `{len(mlb.classes_)}`")
    st.write("**Genres the model knows:**")
    st.write(", ".join(mlb.classes_))
    
    st.markdown("---")
    st.markdown("Made with Streamlit + TensorFlow")

# Main input
description = st.text_area(
    "Enter plot description:",
    height=180,
    placeholder="Example: A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine in order to secure his family's future."
)

# Example buttons
st.markdown("#### 💡 Try these examples:")
col1, col2, col3 = st.columns(3)

example1 = "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine in order to secure his family's future."
example2 = "Decades after their 1984 All Valley Karate Tournament bout, a middle-aged Daniel LaRusso and Johnny Lawrence find themselves martial-arts rivals again."
example3 = "An animated series that follows the exploits of a super scientist and his not-so-bright grandson."

if col1.button("Breaking Bad"):
    description = example1
if col2.button("Cobra Kai"):
    description = example2
if col3.button("Rick and Morty"):
    description = example3

# Predict button
if st.button("🔮 Predict Genres", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Please enter a plot description first.")
    else:
        with st.spinner("Predicting genres..."):
            genres, scores = predict_genres(description, threshold=threshold)

        st.markdown("---")
        st.subheader("Predicted Genres")

        if genres:
            for genre, score in zip(genres, scores):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{genre}**")
                    st.progress(min(float(score), 1.0))
                with col_b:
                    st.metric("Confidence", f"{score*100:.1f}%")
        else:
            st.info("No genres predicted above the threshold. Try lowering the threshold in the sidebar.")

# Footer
st.markdown("---")
st.caption("Multi-label Genre Classification | SimpleRNN | Trained on IMDb plot descriptions")