# Fake News Detection Project

A machine learning project that classifies news articles as real or fake using both traditional NLP techniques and advanced transformer models.

## 🎯 Project Overview

This project implements multiple approaches to detect fake news:
- **Traditional ML**: TF-IDF vectorization with Logistic Regression
- **Deep Learning**: Fine-tuned BERT model for sequence classification

## 📊 Performance Results

### TF-IDF + Logistic Regression Model
- **Accuracy**: 98.62%
- **F1 Score**: 98.67%

#### Detailed Classification Report:
```
               precision    recall  f1-score   support

           0       0.98      0.99      0.99      4284  (Real News)
           1       0.99      0.98      0.99      4696  (Fake News)

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980
```

## 📁 Project Structure

```
FakeNewsDetector/
├── README.md
├── requirements.txt
├── notebooks/
│   └── FakeNewsClassifier_HuggingFace.ipynb
├── scripts/
│   └── train.py
├── models/
│   └── bert-fake-news/  (generated after training)
├── data/
├── app/
└── venv/
```

## 🚀 Quick Start

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

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

## 📚 Dataset

The project uses the `mrm8488/fake-news` dataset from Hugging Face, which contains:
- **Total articles**: ~45,000
- **Training split**: 80% (~36,000 articles)
- **Test split**: 20% (~9,000 articles)
- **Classes**: 
  - 0: Real News
  - 1: Fake News

## 🔧 Models Implemented

### 1. TF-IDF + Logistic Regression
- **Vectorizer**: TF-IDF with 5,000 max features, n-grams (1,2)
- **Classifier**: Logistic Regression with balanced class weights
- **Performance**: 98.62% accuracy

### 2. BERT Fine-tuning
- **Base Model**: `bert-base-uncased`
- **Training**: 3 epochs with evaluation per epoch
- **Optimizer**: AdamW with learning rate 2e-5
- **Batch Size**: 8 per device

## 🛠️ Usage

### Running the Notebook
1. Ensure your virtual environment is activated
2. Start Jupyter: `jupyter notebook`
3. Open `notebooks/FakeNewsClassifier_HuggingFace.ipynb`
4. Make sure the kernel is set to "venv" or "FakeNewsDetector (venv)"
5. Run all cells

### Training BERT Model
```bash
python scripts/train.py
```

The trained model will be saved to `models/bert-fake-news/`

## 📋 Requirements

- Python 3.8+
- pandas
- scikit-learn
- datasets (Hugging Face)
- transformers
- torch
- matplotlib
- seaborn
- jupyter
- ipywidgets

## 🎯 Key Features

- **High Accuracy**: Achieves 98.6% accuracy on test set
- **Multiple Approaches**: Compares traditional ML vs. transformer models
- **Easy Setup**: Simple virtual environment setup
- **Comprehensive Analysis**: Includes confusion matrix and detailed metrics
- **Production Ready**: Trained models can be saved and deployed

## 🔍 Model Analysis

The TF-IDF + Logistic Regression model shows excellent performance:
- **Balanced Performance**: High precision and recall for both classes
- **Low False Positives**: 98% precision for fake news detection
- **Low False Negatives**: 99% recall for real news detection
- **Robust**: Handles class imbalance well with balanced weights

## 🚀 Future Improvements

- [ ] Implement ensemble methods combining multiple models
- [ ] Add cross-validation for more robust evaluation
- [ ] Experiment with other transformer models (RoBERTa, DistilBERT)
- [ ] Deploy model as a web API
- [ ] Add real-time news article classification
- [ ] Implement explainability features (LIME, SHAP)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 📧 Contact

For questions or suggestions, please open an issue or contact the project maintainer.

---

**Note**: This project is for educational and research purposes. Always verify news from multiple reliable sources.
