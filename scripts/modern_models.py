"""
Modern Model Architectures for Fake News Detection
Implements RoBERTa, DeBERTa, and Sentence Transformers for enhanced performance
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
import os
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernFakeNewsDetector:
    """
    Modern fake news detection using state-of-the-art transformer models
    """
    
    def __init__(self, model_name: str = "roberta-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.sentence_model = None
        
    def load_models(self):
        """Load tokenizer and model"""
        logger.info(f"Loading {self.model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        ).to(self.device)
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Models loaded successfully")
        
    def tokenize_data(self, texts: List[str], labels: Optional[List[int]] = None):
        """Tokenize input texts"""
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        if labels is not None:
            tokenized["labels"] = torch.tensor(labels)
            
        return tokenized
    
    def train_model(self, train_texts: List[str], train_labels: List[int], 
                   val_texts: List[str], val_labels: List[int],
                   output_dir: str = "./models/modern_fake_news"):
        """Train the modern model"""
        logger.info("Starting model training...")
        
        # Prepare datasets
        train_dataset = Dataset.from_dict({
            "text": train_texts,
            "label": train_labels
        })
        val_dataset = Dataset.from_dict({
            "text": val_texts,
            "label": val_labels
        })
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """Make predictions on new texts"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_models() first.")
            
        self.model.eval()
        tokenized = self.tokenize_data(texts)
        
        with torch.no_grad():
            outputs = self.model(**tokenized)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(outputs.logits, dim=1)
            
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def semantic_similarity_check(self, query_text: str, reference_texts: List[str], 
                                threshold: float = 0.7) -> Dict:
        """
        Check semantic similarity between query text and reference texts
        Useful for detecting similar fake news patterns
        """
        if self.sentence_model is None:
            raise ValueError("Sentence model not loaded.")
            
        # Encode texts
        query_embedding = self.sentence_model.encode(query_text)
        reference_embeddings = self.sentence_model.encode(reference_texts)
        
        # Calculate similarities
        similarities = util.pytorch_cos_sim(query_embedding, reference_embeddings)[0]
        
        # Find similar texts
        similar_indices = torch.where(similarities > threshold)[0]
        similar_texts = [reference_texts[i] for i in similar_indices]
        similar_scores = [similarities[i].item() for i in similar_indices]
        
        return {
            "similar_texts": similar_texts,
            "similarity_scores": similar_scores,
            "max_similarity": similarities.max().item()
        }

def create_model_comparison():
    """Create comparison of different modern models"""
    models_to_test = [
        "roberta-base",
        "microsoft/deberta-base", 
        "distilbert-base-uncased"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        logger.info(f"Testing {model_name}...")
        detector = ModernFakeNewsDetector(model_name)
        detector.load_models()
        
        # Here you would load your actual data
        # For demonstration, using dummy data
        dummy_texts = ["This is a test news article"] * 10
        dummy_labels = [0, 1] * 5
        
        # Train and evaluate (simplified)
        results[model_name] = {
            "model": detector,
            "status": "loaded"
        }
    
    return results

if __name__ == "__main__":
    # Example usage
    detector = ModernFakeNewsDetector("roberta-base")
    detector.load_models()
    
    # Test prediction
    test_texts = [
        "Scientists discover new breakthrough in renewable energy",
        "Aliens spotted in downtown area, government covering it up"
    ]
    
    predictions, probabilities = detector.predict(test_texts)
    
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        label = "Fake" if pred == 0 else "Real"
        confidence = max(prob)
        print(f"Text: {text}")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print()
