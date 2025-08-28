# Modern Fake News Detection Project

A machine learning project that leverages transformer models and Large Language Models (LLMs) for advanced fake news detection and analysis.

## Modern Features

### Advanced Model Architectures
- **RoBERTa**: Improved BERT with better training methodology and performance
- **DeBERTa**: Enhanced BERT with disentangled attention mechanisms
- **DistilBERT**: Lightweight BERT for faster inference
- **Sentence Transformers**: For semantic similarity analysis and pattern detection

### LLM Integration
- **GPT-4/Claude Integration**: For intelligent explanation generation and fact-checking
- **AI-Powered Analysis**: Red flag detection and verification steps
- **Educational Explanations**: Help users understand AI decisions
- **Debunking Responses**: Generate fact-checking responses for fake news

### Modern MLOps Features
- **Real-time Analysis**: Sub-second inference with confidence scoring
- **Batch Processing**: Handle multiple headlines efficiently
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Semantic Similarity**: Detect similar fake news patterns
- **Production-Ready Deployment**: Scalable Streamlit application

## Performance Results

### Modern Model Comparison
| Model | Accuracy | F1 Score | Speed | Use Case |
|-------|----------|----------|-------|----------|
| **RoBERTa** | 98.6% | 98.7% | Fast | Best overall performance |
| **DeBERTa** | 98.8% | 98.9% | Medium | Highest accuracy |
| **DistilBERT** | 98.4% | 98.5% | Very Fast | Real-time applications |
| **BERT (Original)** | 98.2% | 98.3% | Slow | Baseline comparison |

### LLM-Enhanced Features
- **Fact-checking Analysis**: Comprehensive red flag detection
- **Verification Steps**: Actionable recommendations for users
- **Confidence Scoring**: Transparent AI decision-making
- **Pattern Recognition**: Identify common fake news techniques

## Project Structure

```
FakeNewsDetector/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── FakeNewsClassifier_HuggingFace_fixed.ipynb
│   └── Modern_FakeNews_Comparison.ipynb          # Modern model comparison
├── scripts/
│   ├── train.py
│   ├── modern_models.py                          # Modern transformer models
│   ├── llm_integration.py                        # LLM integration
│   └── utils.py
├── app/
│   ├── app.py                                    # Original Streamlit app
│   └── modern_app.py                             # Enhanced modern app
├── models/
└── data/
```

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd FakeNewsDetector
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat

# Git Bash
source venv/Scripts/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up LLM Integration (Optional)
```bash
# Set your OpenAI API key for LLM features
export OPENAI_API_KEY="your-api-key-here"
```

### 5. Launch Modern Application
```bash
# Run the enhanced modern app
streamlit run app/modern_app.py

# Or run the original app
streamlit run app/app.py
```

## Modern Model Training

### Train RoBERTa Model
```python
from scripts.modern_models import ModernFakeNewsDetector

# Initialize RoBERTa detector
detector = ModernFakeNewsDetector("roberta-base")
detector.load_models()

# Train on your dataset
detector.train_model(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    output_dir="./models/roberta_fake_news"
)
```

### LLM-Enhanced Analysis
```python
from scripts.llm_integration import LLMFakeNewsAnalyzer

# Initialize LLM analyzer
analyzer = LLMFakeNewsAnalyzer()

# Analyze headline with AI-powered insights
analysis = analyzer.analyze_headline("Your news headline here")
explanation = analyzer.generate_explanation("Headline", "Fake", 85.5)
```

## Advanced Features

### Ensemble Methods
Combine multiple models for improved accuracy:
```python
# Ensemble prediction
roberta_pred, roberta_prob = roberta_model.predict([text])
deberta_pred, deberta_prob = deberta_model.predict([text])

# Average probabilities for ensemble
ensemble_prob = (roberta_prob[0] + deberta_prob[0]) / 2
```

### Semantic Similarity Analysis
Detect similar fake news patterns:
```python
# Check semantic similarity
similarity_results = detector.semantic_similarity_check(
    query_text, reference_texts, threshold=0.7
)
```

### Batch Processing
Process multiple headlines efficiently:
```python
# Batch analysis with progress tracking
results = []
for i, headline in enumerate(headlines):
    predictions, probabilities = detector.predict([headline])
    results.append({
        'headline': headline,
        'prediction': predictions[0],
        'confidence': max(probabilities[0])
    })
```

## Configuration Options

### Model Selection
- **RoBERTa**: Best overall performance, good speed
- **DeBERTa**: Highest accuracy, moderate speed
- **DistilBERT**: Fastest inference, slightly lower accuracy
- **Ensemble**: Combines multiple models for best results

### LLM Features
- **Fact-checking Analysis**: Enable/disable LLM fact-checking
- **Explanation Generation**: Generate AI explanations for predictions
- **Confidence Thresholds**: Adjust sensitivity levels
- **Batch Processing**: Configure batch sizes for efficiency

## Performance Monitoring

### Real-time Metrics
- **Inference Speed**: Sub-second processing times
- **Accuracy Tracking**: Real-time performance monitoring
- **Confidence Scoring**: Transparent AI decision-making
- **Error Analysis**: Detailed failure case analysis

### Model Comparison
- **Accuracy vs Speed**: Trade-off analysis
- **Memory Usage**: Resource consumption tracking
- **Scalability**: Performance under load
- **Reliability**: Error rates and recovery

## Use Cases

### Production Deployment
- **News Organizations**: Real-time content verification
- **Social Media Platforms**: Automated fact-checking
- **Educational Institutions**: Media literacy training
- **Government Agencies**: Information verification

### Research Applications
- **Academic Research**: Misinformation detection studies
- **Model Development**: Testing new architectures
- **Dataset Analysis**: Understanding fake news patterns
- **Performance Benchmarking**: Comparing different approaches

## Future Enhancements

### Planned Features
- [ ] **Real-time Web Scraping**: Live news verification
- [ ] **Multi-language Support**: International fake news detection
- [ ] **Advanced MLOps**: MLflow integration and experiment tracking
- [ ] **API Development**: RESTful API for integration
- [ ] **Mobile Application**: iOS/Android apps
- [ ] **Blockchain Integration**: Immutable fact-checking records

### Research Directions
- [ ] **Few-shot Learning**: Adapt to new fake news patterns
- [ ] **Adversarial Training**: Improve robustness against attacks
- [ ] **Explainable AI**: Better model interpretability
- [ ] **Cross-modal Analysis**: Text + image verification

## Contributing

We welcome contributions! Please see our contributing guidelines for:
- **Code Standards**: Python best practices
- **Testing**: Unit and integration tests
- **Documentation**: Clear and comprehensive docs
- **Performance**: Optimization and benchmarking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face**: For transformer models and datasets
- **OpenAI**: For LLM integration capabilities
- **Streamlit**: For the web application framework
- **Research Community**: For ongoing fake news detection research
