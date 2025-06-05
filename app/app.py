import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Hugging Face model path (change this to your actual repo ID)
MODEL_DIR = "ragkasi/bert-fake-news"

@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

classifier = load_pipeline()

# UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news **headline** or **statement**, and this app will predict if it's **real** or **fake**.")

news_input = st.text_area("✏️ News Text", height=150)

if st.button("🔍 Check News"):
    if news_input.strip():
        result = classifier(news_input)[0]
        label = result["label"]
        score = result["score"]

        # Adjust label display
        if label == "LABEL_1":
            st.error(f"🚨 Likely **Fake News** (Confidence: `{score:.2f}`)")
        else:
            st.success(f"✅ Likely **Real News** (Confidence: `{score:.2f}`)")
    else:
        st.warning("⚠️ Please enter a news statement.")
