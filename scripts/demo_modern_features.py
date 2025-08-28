#!/usr/bin/env python3
"""
Modern Fake News Detection Demo
Demonstrates the latest features including LLM integration and modern model architectures
"""

import os
import sys
import time
from typing import List, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modern_models import ModernFakeNewsDetector
from llm_integration import LLMFakeNewsAnalyzer, HybridFakeNewsDetector

def demo_modern_models():
    """Demonstrate modern transformer models"""
    print("Modern Model Architecture Demo")
    print("=" * 50)
    
    # Test headlines
    test_headlines = [
        "Scientists discover miracle cure for all diseases",
        "New study shows coffee prevents cancer",
        "Government announces new economic policy",
        "Aliens spotted in downtown area, government covering it up",
        "Breaking: Major breakthrough in renewable energy technology"
    ]
    
    # Initialize different models
    models = {
        'RoBERTa': ModernFakeNewsDetector('roberta-base'),
        'DeBERTa': ModernFakeNewsDetector('microsoft/deberta-base'),
        'DistilBERT': ModernFakeNewsDetector('distilbert-base-uncased')
    }
    
    print("Loading models...")
    for name, model in models.items():
        print(f"Loading {name}...")
        model.load_models()
        print(f"{name} loaded successfully")
    
    # Test each model
    for model_name, model in models.items():
        print(f"\nTesting {model_name}:")
        print("-" * 30)
        
        start_time = time.time()
        predictions, probabilities = model.predict(test_headlines)
        inference_time = time.time() - start_time
        
        for i, (headline, pred, prob) in enumerate(zip(test_headlines, predictions, probabilities)):
            label = "Fake" if pred == 0 else "Real"
            confidence = max(prob) * 100
            print(f"{i+1}. {headline[:50]}...")
            print(f"   Prediction: {label} ({confidence:.1f}% confidence)")
        
        print(f"   ⚡ Inference time: {inference_time:.2f}s")
    
    return models

def demo_llm_integration():
    """Demonstrate LLM integration features"""
    print("\nLLM Integration Demo")
    print("=" * 50)
    
    # Initialize LLM analyzer
    try:
        llm_analyzer = LLMFakeNewsAnalyzer()
        print("LLM analyzer initialized")
    except Exception as e:
        print(f"LLM analyzer not available: {e}")
        print("To enable LLM features, set your OPENAI_API_KEY environment variable")
        return None
    
    # Test headlines for LLM analysis
    test_headlines = [
        "Scientists discover miracle cure for all diseases",
        "Government announces new economic policy"
    ]
    
    for headline in test_headlines:
        print(f"\nAnalyzing: {headline}")
        print("-" * 40)
        
        # Get LLM analysis
        analysis = llm_analyzer.analyze_headline(headline)
        
        if 'error' not in analysis:
            if 'analysis' in analysis:
                print(f"Analysis: {analysis['analysis']}")
            
            if 'red_flags' in analysis and analysis['red_flags']:
                print(f"Red Flags: {', '.join(analysis['red_flags'])}")
            
            if 'confidence' in analysis:
                print(f"Confidence: {analysis['confidence']}")
            
            if 'verification_steps' in analysis:
                print("Verification Steps:")
                for i, step in enumerate(analysis['verification_steps'], 1):
                    print(f"   {i}. {step}")
        else:
            print(f"Analysis failed: {analysis['error']}")
    
    return llm_analyzer

def demo_hybrid_analysis():
    """Demonstrate hybrid ML + LLM analysis"""
    print("\nHybrid Analysis Demo")
    print("=" * 50)
    
    # Initialize components
    try:
        ml_model = ModernFakeNewsDetector('roberta-base')
        ml_model.load_models()
        
        llm_analyzer = LLMFakeNewsAnalyzer()
        
        hybrid_detector = HybridFakeNewsDetector(ml_model, llm_analyzer)
        print("Hybrid detector initialized")
    except Exception as e:
        print(f"Failed to initialize hybrid detector: {e}")
        return
    
    # Test hybrid analysis
    test_headline = "Scientists discover miracle cure for all diseases"
    
    print(f"\nHybrid Analysis: {test_headline}")
    print("-" * 50)
    
    result = hybrid_detector.analyze(test_headline)
    
    print(f"ML Prediction: {result['ml_prediction']}")
    print(f"ML Confidence: {result['ml_confidence']:.1f}%")
    print(f"AI Explanation: {result['explanation']}")
    
    if 'llm_analysis' in result and 'analysis' in result['llm_analysis']:
        print(f"LLM Analysis: {result['llm_analysis']['analysis']}")

def demo_semantic_similarity():
    """Demonstrate semantic similarity analysis"""
    print("\nSemantic Similarity Demo")
    print("=" * 50)
    
    # Initialize model
    detector = ModernFakeNewsDetector('roberta-base')
    detector.load_models()
    
    # Test semantic similarity
    query_text = "Scientists discover miracle cure for all diseases"
    reference_texts = [
        "Doctors hate this one trick",
        "Miracle cure found in ancient herb",
        "New medical breakthrough announced",
        "Government announces new economic policy",
        "Weather forecast for tomorrow"
    ]
    
    print(f"Query: {query_text}")
    print("\nChecking similarity against reference texts...")
    
    similarity_results = detector.semantic_similarity_check(
        query_text, reference_texts, threshold=0.5
    )
    
    print(f"\nSimilarity Results:")
    for text, score in zip(similarity_results['similar_texts'], similarity_results['similarity_scores']):
        print(f"• {text} (similarity: {score:.3f})")
    
    print(f"\nMaximum similarity: {similarity_results['max_similarity']:.3f}")

def demo_ensemble_methods():
    """Demonstrate ensemble methods"""
    print("\nEnsemble Methods Demo")
    print("=" * 50)
    
    # Initialize multiple models
    models = {
        'RoBERTa': ModernFakeNewsDetector('roberta-base'),
        'DeBERTa': ModernFakeNewsDetector('microsoft/deberta-base')
    }
    
    for name, model in models.items():
        model.load_models()
    
    # Test ensemble prediction
    test_text = "Scientists discover miracle cure for all diseases"
    
    print(f"Testing ensemble prediction for: {test_text}")
    print("-" * 50)
    
    # Get individual predictions
    roberta_pred, roberta_prob = models['RoBERTa'].predict([test_text])
    deberta_pred, deberta_prob = models['DeBERTa'].predict([test_text])
    
    print(f"RoBERTa: {'Fake' if roberta_pred[0] == 0 else 'Real'} ({max(roberta_prob[0])*100:.1f}%)")
    print(f"DeBERTa: {'Fake' if deberta_pred[0] == 0 else 'Real'} ({max(deberta_prob[0])*100:.1f}%)")
    
    # Ensemble (average probabilities)
    ensemble_prob = (roberta_prob[0] + deberta_prob[0]) / 2
    ensemble_pred = 0 if ensemble_prob[0] > ensemble_prob[1] else 1
    
    print(f"Ensemble: {'Fake' if ensemble_pred == 0 else 'Real'} ({max(ensemble_prob)*100:.1f}%)")

def main():
    """Run all demos"""
    print("Modern Fake News Detection - Feature Demo")
    print("=" * 60)
    print("This demo showcases the latest advancements in fake news detection")
    print("including modern transformer models and LLM integration.\n")
    
    try:
        # Demo 1: Modern Models
        models = demo_modern_models()
        
        # Demo 2: LLM Integration
        llm_analyzer = demo_llm_integration()
        
        # Demo 3: Hybrid Analysis
        demo_hybrid_analysis()
        
        # Demo 4: Semantic Similarity
        demo_semantic_similarity()
        
        # Demo 5: Ensemble Methods
        demo_ensemble_methods()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Modern transformer models (RoBERTa, DeBERTa, DistilBERT)")
        print("• LLM integration for explanation generation")
        print("• Hybrid ML + LLM analysis")
        print("• Semantic similarity detection")
        print("• Ensemble methods for improved accuracy")
        
        print("\nTo run the full application:")
        print("streamlit run app/modern_app.py")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Please check your setup and dependencies.")

if __name__ == "__main__":
    main()
