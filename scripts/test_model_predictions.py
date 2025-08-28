#!/usr/bin/env python3
"""
Test script to debug model predictions and understand classification issues.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def test_model_predictions():
    """Test the model with various inputs to debug classification issues"""
    
    # Load model
    model_path = "models/deployment/bert-fake-news"
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        
        print("✓ Model loaded successfully!")
        
        # Test cases
        test_cases = [
            "The weather is nice outside",
            "Aliens in my basement", 
            "Breaking: Stock market crashes 50% today",
            "Scientists discover cure for cancer",
            "Donald Trump announces new tax policy",
            "Local cat wins beauty contest",
            "COVID-19 vaccine proven 95% effective in trials",
            "Unicorns found living in Central Park",
            "The president will visit France next week",
            "My neighbor borrowed my lawnmower"
        ]
        
        print("\n" + "="*80)
        print("TESTING MODEL PREDICTIONS")
        print("="*80)
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: '{text}'")
            print("-" * 60)
            
            # Get raw prediction
            results = classifier(text)
            print(f"Raw results: {results}")
            
            # Extract scores
            scores = results[0] if isinstance(results[0], list) else results
            
            label_0_score = 0
            label_1_score = 0
            
            for item in scores:
                if item.get('label') == 'LABEL_0':
                    label_0_score = item.get('score', 0)
                elif item.get('label') == 'LABEL_1':
                    label_1_score = item.get('score', 0)
            
            print(f"LABEL_0 (Fake): {label_0_score:.3f}")
            print(f"LABEL_1 (Real): {label_1_score:.3f}")
            
            # Current logic
            if label_1_score > label_0_score:
                prediction = "REAL NEWS"
                confidence = label_1_score
            else:
                prediction = "FAKE NEWS"
                confidence = label_0_score
                
            print(f"Prediction: {prediction} (Confidence: {confidence:.1%})")
            
            # Check if this makes sense
            is_reasonable = analyze_prediction(text, prediction, confidence)
            print(f"Analysis: {'✓ REASONABLE' if is_reasonable else '✗ PROBLEMATIC'}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def analyze_prediction(text, prediction, confidence):
    """Analyze if a prediction makes sense"""
    
    # Simple heuristics for what should be considered reasonable
    text_lower = text.lower()
    
    # Clearly fake/absurd statements
    absurd_keywords = ["aliens", "unicorns", "basement", "magic", "dragon"]
    if any(keyword in text_lower for keyword in absurd_keywords):
        return prediction == "FAKE NEWS"
    
    # Normal statements that aren't really "news"
    normal_keywords = ["weather", "nice", "neighbor", "borrowed", "lawnmower"]
    if any(keyword in text_lower for keyword in normal_keywords):
        # These shouldn't be classified as news at all, but if forced...
        # Could go either way depending on context
        return True  # We'll accept any prediction for these
    
    # Real news-like statements
    news_keywords = ["breaking", "announces", "president", "scientists", "discover"]
    if any(keyword in text_lower for keyword in news_keywords):
        # These could be real or fake depending on verification
        return True  # Accept any prediction with reasonable confidence
    
    return True  # Default to accepting prediction

if __name__ == "__main__":
    test_model_predictions() 