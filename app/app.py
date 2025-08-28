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
    "./models/deployment/bert-fake-news",                                      # Deployment-ready model (PRIORITY)
    "./models/bert-fake-news/iteration2/bert-fake-news/checkpoint-5000",       # Earlier checkpoint (best balance)
    "./models/bert-fake-news/iteration2/bert-fake-news/checkpoint-10000",      # Middle checkpoint  
    "./models/bert-fake-news/iteration2/bert-fake-news/checkpoint-15000",      # Latest checkpoint
    "../models/deployment/bert-fake-news",                                     # Alternative deployment path
    "../models/bert-fake-news/iteration2/bert-fake-news/checkpoint-5000",      # Alternative path
    "../models/bert-fake-news/iteration2/bert-fake-news/checkpoint-10000",     # Alternative path
    "../models/bert-fake-news/iteration2/bert-fake-news/checkpoint-15000",     # Alternative path
]

def check_if_news_like(text):
    """Check if the input text looks like news content"""
    text_lower = text.lower().strip()
    
    # Too short to be meaningful
    if len(text.split()) < 3:
        return False
    
    # Obviously personal/casual statements (very specific)
    personal_starts = [
        "my neighbor", "my friend", "my family", "my dog", "my cat",
        "i went", "i saw", "i think", "i believe", "i feel",
        "the weather is", "it's raining", "it's sunny", "it's cold"
    ]
    
    # Obviously fictional/absurd content
    fictional_indicators = [
        "aliens in my", "unicorns", "dragons", "magic", "wizard", "fairy",
        "santa claus", "easter bunny", "tooth fairy"
    ]
    
    # Check for obviously personal statements
    if any(text_lower.startswith(start) for start in personal_starts):
        return False
        
    # Check for obviously fictional content
    if any(indicator in text_lower for indicator in fictional_indicators):
        return False
    
    # If it contains news-like words, definitely news
    news_indicators = [
        "breaking", "report", "according", "announce", "discover", "study", 
        "government", "president", "minister", "official", "research", 
        "scientist", "expert", "source", "investigate", "election",
        "economy", "market", "stock", "company", "court", "judge",
        "hospital", "doctor", "vaccine", "covid", "crisis", "emergency"
    ]
    
    if any(indicator in text_lower for indicator in news_indicators):
        return True
    
    # If it's structured like a headline or formal statement, likely news
    if len(text.split()) >= 5:
        return True
        
    # Default to treating it as news-like (be permissive)
    return True

def apply_bias_correction(text, label_0_score, label_1_score):
    """
    Apply bias correction to handle model's complex bias patterns
    Returns: (prediction_type, confidence)
    """
    text_lower = text.lower().strip()
    
    # Indicators that suggest potential fake news (be more selective)
    strong_fake_indicators = [
        "you won't believe", "doctors hate this", "this one trick",
        "miracle cure", "secret they don't want you to know",
        "amazing discovery that will change everything"
    ]
    
    # Sensational but potentially legitimate news language
    sensational_but_news = [
        "breaking:", "urgent:", "major", "significant", "important"
    ]
    
    # Check for obviously fake patterns
    has_strong_fake = any(indicator in text_lower for indicator in strong_fake_indicators)
    has_sensational_news = any(indicator in text_lower for indicator in sensational_but_news)
    
    # Model tends to over-classify legitimate news as fake, so be more generous with "real" classifications
    
    # If model predicts fake news with very high confidence AND content has strong fake indicators, trust it
    if label_0_score > 0.95 and has_strong_fake:
        return "Fake News", label_0_score
    
    # If model predicts fake news with high confidence but content looks like legitimate breaking news, be more cautious
    if label_0_score > 0.90 and has_sensational_news:
        # Reduce confidence in fake prediction for breaking news
        return "Fake News", max(0.60, label_0_score - 0.20)
    
    # If model predicts real news for obviously absurd content, flip to fake
    absurd_indicators = ["unicorns", "aliens", "magic", "wizard", "dragon"]
    has_absurd = any(indicator in text_lower for indicator in absurd_indicators)
    
    if label_1_score > 0.80 and has_absurd:
        return "Fake News", 0.75
    
    # For normal cases, trust the model more but with slight adjustments
    if label_1_score > label_0_score:
        # Model predicts real news - be less aggressive in reducing confidence
        if label_1_score > 0.95:
            adjusted_confidence = label_1_score - 0.05  # Minor reduction for overconfidence
        else:
            adjusted_confidence = label_1_score  # Trust the model
        
        return "Real News", max(0.50, adjusted_confidence)
    else:
        # Model predicts fake news - be more cautious since model tends to over-predict fake
        if label_0_score > 0.90:
            # High confidence fake predictions, reduce slightly
            adjusted_confidence = label_0_score - 0.10
        else:
            adjusted_confidence = label_0_score
            
        return "Fake News", max(0.50, adjusted_confidence)

@st.cache_resource
def load_model():
    """Load the model from available paths with better error handling"""
    
    for model_path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                st.info(f"Attempting to load model from: `{model_path}`")
                
                # Check if required files exist
                config_file = os.path.join(model_path, "config.json")
                pytorch_model_file = os.path.join(model_path, "pytorch_model.bin")
                safetensors_model_file = os.path.join(model_path, "model.safetensors")
                
                # Check for either pytorch_model.bin or model.safetensors
                has_model_file = os.path.exists(pytorch_model_file) or os.path.exists(safetensors_model_file)
                
                if os.path.exists(config_file) and has_model_file:
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
                    
                    # DEBUG: Show raw results for troubleshooting
                    st.expander("Debug Info").write(f"Raw model output: {results}")
                    
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
                    
                    # BIAS CORRECTION: The model is heavily biased toward "real news"
                    # Apply bias correction based on content analysis and confidence patterns
                    
                    # Check if input looks like actual news first
                    is_news_like = check_if_news_like(news)
                    
                    if not is_news_like:
                        st.warning("**Not News Content**")
                        st.info("This doesn't appear to be a news statement. The model is designed for news articles and headlines.")
                        st.write(f"**Analysis for reference only:**")
                        
                        # For non-news content, don't trust high "real news" predictions
                        if label_1_score > 0.85:
                            st.warning("**Note**: Model incorrectly classifies non-news content as 'real news' due to training bias.")
                    else:
                        # BIAS-AWARE PREDICTION LOGIC
                        confidence_gap = abs(label_1_score - label_0_score)
                        
                        # If the gap is very small, the model is uncertain (reduced threshold)
                        if confidence_gap < 0.05:  # Reduced from 0.1 to 0.05
                            st.warning("**Highly Uncertain Classification**")
                            st.info(f"Model confidence is extremely low. LABEL_0: {label_0_score:.1%}, LABEL_1: {label_1_score:.1%}")
                            st.warning("Cannot make reliable prediction - results too close.")
                        else:
                            # Apply content-aware bias correction
                            prediction_type, confidence = apply_bias_correction(news, label_0_score, label_1_score)
                            
                            # Display results with more nuanced confidence levels
                            if prediction_type == "Real News":
                                if confidence > 0.90:
                                    st.success(f"**Likely Real News**\n\nConfidence: {confidence:.1%}")
                                elif confidence > 0.70:
                                    st.success(f"**Probably Real News**\n\nConfidence: {confidence:.1%}")
                                    st.info("**Moderate confidence** - Consider verifying through other sources")
                                elif confidence > 0.55:
                                    st.warning(f"**Leaning Real News**\n\nConfidence: {confidence:.1%}")
                                    st.info("**Low confidence** - Model is uncertain, please verify!")
                                else:
                                    st.info(f"**Possibly Real News**\n\nConfidence: {confidence:.1%}")
                                    st.warning("**Very low confidence** - Results inconclusive")
                            else:
                                if confidence > 0.90:
                                    st.error(f"**Likely Fake News**\n\nConfidence: {confidence:.1%}")
                                elif confidence > 0.70:
                                    st.error(f"**Probably Fake News**\n\nConfidence: {confidence:.1%}")
                                    st.info("**Moderate confidence** - Consider fact-checking")
                                elif confidence > 0.55:
                                    st.warning(f"**Leaning Fake News**\n\nConfidence: {confidence:.1%}")
                                    st.info("**Low confidence** - Model is uncertain, please verify!")
                                else:
                                    st.info(f"**Possibly Fake News**\n\nConfidence: {confidence:.1%}")
                                    st.warning("**Very low confidence** - Results inconclusive")
                            
                            # Show bias correction info
                            st.info("**Bias-aware prediction**: Adjustments applied to account for model training patterns.")
                    
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
        ### Model Status:
        
        **Fixed Issues:**
        - Now correctly identifies non-news content
        - Filters out personal statements and fictional content
        - Added debug information for transparency
        
        **Known Limitations:**
        - Model has training bias (tends to over-classify as fake)
        - Works best with clear news headlines  
        - May struggle with nuanced or ambiguous content
        - Requires retraining for optimal performance
        
        ### Usage Tips:
        - Enter complete news headlines or articles
        - Look for the "Non-News Content" warning for non-news input
        - Check the debug info to understand model confidence
        - Always verify important news through multiple sources
        
        ### Best Results With:
        - Political news and announcements
        - Business and economic news
        - Health and scientific research news
        - Breaking news headlines
        """)
        
        st.markdown("""
        ---
        **Technical Details:**
        - Architecture: BERT for Sequence Classification
        - Training: Fine-tuned on multiple news datasets
        - Labels: LABEL_0 = Fake News, LABEL_1 = Real News
        - Version: Includes content filtering and bias awareness
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
