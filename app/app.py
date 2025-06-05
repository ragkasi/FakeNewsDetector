import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Fix for PyTorch/Streamlit compatibility issue
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Set Streamlit page config
st.set_page_config(page_title="Fake News Detector")

# Model Path Configuration
# Try different possible model locations
POSSIBLE_MODEL_PATHS = [
    "./models/bert-fake-news/iteration2/bert-fake-news/checkpoint-5000",   # Earlier checkpoint (best balance)
    "./models/bert-fake-news/iteration2/bert-fake-news/checkpoint-10000",  # Middle checkpoint  
    "./models/bert-fake-news/iteration2/bert-fake-news/checkpoint-15000",  # Latest checkpoint
    "../models/bert-fake-news/iteration2/bert-fake-news/checkpoint-5000",  # Alternative path
]

@st.cache_resource
def load_model():
    """Load the model from available paths with better error handling"""
    
    for model_path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                st.info(f"Attempting to load model from: `{model_path}`")
                
                # Check if required files exist
                config_file = os.path.join(model_path, "config.json")
                model_file = os.path.join(model_path, "model.safetensors")
                
                if os.path.exists(config_file) and os.path.exists(model_file):
                    # Try to load tokenizer from model path first, fallback to base BERT
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                    except:
                        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
                    
                    st.success(f"Model loaded successfully from: `{model_path}`")
                    return classifier
                else:
                    st.warning(f"Model files missing in: `{model_path}`")
                    continue
                    
            except Exception as e:
                st.warning(f"Failed to load from `{model_path}`: {str(e)}")
                continue
    
    # If no model found, show helpful error message
    st.error("**No valid model found!**")
    st.markdown("""
    ### Troubleshooting:
    
    **Option 1: Copy your trained model**
    - Copy your trained model files to: `./models/bert-fake-news/`
    - Required files: `config.json`, `model.safetensors`, and tokenizer files
    
    **Option 2: Use checkpoint directly**
    - If you have checkpoints, copy the contents of your latest checkpoint to `./models/bert-fake-news/`
    
    **Option 3: Retrain the model**
    - Use the provided training notebook to train a new model
    - Download and extract the model to the correct location
    
    ### Expected Model Structure:
    ```
    models/
    └── bert-fake-news/
        ├── config.json
        ├── model.safetensors
        └── tokenizer files
    ```
    """)
    return None

# Load Model
classifier = load_model()

# App UI
st.title("Fake News Detector")
st.markdown("Enter a news **headline** or **statement** below, and this app will tell you whether it's likely **real** or **fake** using a fine-tuned BERT model.")

# Only show the input if model is loaded
if classifier is not None:
    news = st.text_area("Enter News Text:", height=150)

    if st.button("Check News"):
        if news.strip():
            with st.spinner("Analyzing..."):
                try:
                    results = classifier(news)
                    
                    # Extract scores from the nested list format
                    scores = results[0] if isinstance(results[0], list) else results
                    
                    # Get individual scores
                    label_0_score = 0
                    label_1_score = 0
                    
                    for item in scores:
                        if item.get('label') == 'LABEL_0':
                            label_0_score = item.get('score', 0)
                        elif item.get('label') == 'LABEL_1':
                            label_1_score = item.get('score', 0)
                    
                    # Final prediction based on correct label mapping
                    # LABEL_1 = Real News, LABEL_0 = Fake News
                    if label_1_score > label_0_score:
                        confidence = label_1_score
                        if confidence > 0.95:
                            st.success(f"**Likely Real News**\n\nConfidence: {confidence:.1%}")
                        elif confidence > 0.75:
                            st.success(f"**Probably Real News**\n\nConfidence: {confidence:.1%}")
                            st.info("**Moderate confidence** - Consider verifying through other sources")
                        else:
                            st.warning(f"**Leaning Real News**\n\nConfidence: {confidence:.1%}")
                            st.warning("**Low confidence** - Model is uncertain, please verify!")
                    else:
                        confidence = label_0_score
                        if confidence > 0.95:
                            st.error(f"**Likely Fake News**\n\nConfidence: {confidence:.1%}")
                        elif confidence > 0.75:
                            st.error(f"**Probably Fake News**\n\nConfidence: {confidence:.1%}")
                            st.info("**Moderate confidence** - Consider fact-checking")
                        else:
                            st.warning(f"**Leaning Fake News**\n\nConfidence: {confidence:.1%}")
                            st.warning("**Low confidence** - Model is uncertain, please verify!")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a news statement.")
else:
    st.info("Please fix the model loading issue above to use the detector.")

# Model Info Sidebar
with st.sidebar:
    st.header("Model Information")
    
    if classifier is not None:
        st.success("Model Status: **Loaded**")
        st.markdown("""
        **Model Details:**
        - Architecture: BERT for Sequence Classification
        - Labels: Real (LABEL_0) / Fake (LABEL_1)
        - Training: Fine-tuned on news datasets
        """)
    else:
        st.error("Model Status: **Not Loaded**")
        
    st.markdown("""
    ### Usage Tips:
    - Enter complete news headlines or articles
    - Longer, more detailed text generally gives better results
    - The model works best with English news content
    
    ### Limitations:
    - This is a machine learning model and may make mistakes
    - Always verify important news through multiple sources
    - Consider the confidence score when interpreting results
    """)
