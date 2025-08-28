"""
LLM Integration for Enhanced Fake News Detection
Integrates GPT-4/Claude for explanation generation and fact-checking
"""

import openai
import os
from typing import Dict, List, Optional, Tuple
import json
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMFakeNewsAnalyzer:
    """
    Enhanced fake news detection using Large Language Models
    Provides explanations, fact-checking, and reasoning
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            try:
                # Clear ALL proxy-related environment variables that might cause issues
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
                old_proxies = {}
                for var in proxy_vars:
                    old_proxies[var] = os.environ.pop(var, None)
                
                # Also clear any OpenAI-specific environment variables that might interfere
                openai_vars = ['OPENAI_PROXY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']
                old_openai_vars = {}
                for var in openai_vars:
                    old_openai_vars[var] = os.environ.pop(var, None)
                
                # Initialize with minimal configuration
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    timeout=30.0
                )
                
                # Restore environment variables
                for var, value in old_proxies.items():
                    if value is not None:
                        os.environ[var] = value
                for var, value in old_openai_vars.items():
                    if value is not None:
                        os.environ[var] = value
                        
                logger.info("OpenAI client initialized successfully")
                    
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                # Try with even more minimal configuration
                try:
                    logger.info("Trying minimal OpenAI client initialization...")
                    # Clear environment completely for this attempt
                    import os
                    original_env = dict(os.environ)
                    os.environ.clear()
                    os.environ['OPENAI_API_KEY'] = self.api_key
                    
                    self.client = openai.OpenAI()
                    
                    # Restore original environment
                    os.environ.clear()
                    os.environ.update(original_env)
                    
                    logger.info("Minimal initialization successful")
                except Exception as e2:
                    logger.error(f"Minimal initialization also failed: {e2}")
                    self.client = None
        else:
            logger.warning("No OpenAI API key provided. LLM features will be disabled.")
            self.client = None
            
        # Predefined prompts for different analysis types
        self.prompts = {
            "fact_check": """
            Analyze the following news headline for potential fake news indicators:
            
            Headline: {headline}
            
            Please provide:
            1. Fact-checking analysis (1-3 sentences)
            2. Red flags or suspicious elements
            3. Confidence level (High/Medium/Low)
            4. Recommended verification steps
            
            Format your response as JSON:
            {{
                "analysis": "fact-checking analysis",
                "red_flags": ["flag1", "flag2"],
                "confidence": "High/Medium/Low",
                "verification_steps": ["step1", "step2"]
            }}
            """,
            
            "explanation": """
            Explain why this news headline might be classified as {classification}:
            
            Headline: {headline}
            Classification: {classification}
            Confidence: {confidence}%
            
            Provide a clear, educational explanation that helps users understand:
            1. Key indicators that led to this classification
            2. Common patterns in {classification.lower()} news
            3. How to verify this information independently
            
            Keep the explanation under 150 words and use simple language.
            """,
            
            "debunk": """
            Create a fact-checking response for this potentially fake news:
            
            Headline: {headline}
            
            Provide a debunking response that:
            1. Identifies the false claims
            2. Provides factual corrections
            3. Suggests reliable sources for verification
            4. Explains why this type of misinformation is harmful
            
            Format as a concise, factual response suitable for social media sharing.
            """,
            
            "enhanced_analysis": """
            Perform a comprehensive analysis of this news headline:
            
            Headline: {headline}
            
            Analyze for:
            1. Emotional manipulation techniques
            2. Logical fallacies
            3. Source credibility indicators
            4. Factual accuracy assessment
            5. Potential bias indicators
            
            Provide a structured analysis with specific examples from the text.
            """
        }
    
    def analyze_headline(self, headline: str) -> Dict:
        """
        Comprehensive analysis of a news headline using LLM
        """
        if not self.client:
            return {"error": "No API key available"}
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert fact-checker and media literacy specialist."},
                    {"role": "user", "content": self.prompts["fact_check"].format(headline=headline)}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"analysis": content, "error": "Failed to parse JSON response"}
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {"error": str(e)}
    
    def generate_explanation(self, headline: str, classification: str, confidence: float) -> str:
        """
        Generate human-readable explanation for classification
        """
        if not self.client:
            return "LLM explanation not available (no API key)"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that explains AI classifications in simple terms."},
                    {"role": "user", "content": self.prompts["explanation"].format(
                        headline=headline,
                        classification=classification,
                        confidence=confidence
                    )}
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Explanation generation failed: {str(e)}"
    
    def create_debunking_response(self, headline: str) -> str:
        """
        Create a debunking response for fake news
        """
        if not self.client:
            return "Debunking response not available (no API key)"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional fact-checker."},
                    {"role": "user", "content": self.prompts["debunk"].format(headline=headline)}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error creating debunking response: {e}")
            return f"Debunking response generation failed: {str(e)}"
    
    def enhanced_analysis(self, headline: str) -> Dict:
        """
        Perform enhanced analysis with multiple aspects
        """
        if not self.client:
            return {"error": "No API key available"}
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in media literacy and misinformation detection."},
                    {"role": "user", "content": self.prompts["enhanced_analysis"].format(headline=headline)}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            return {"analysis": response.choices[0].message.content}
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            return {"error": str(e)}
    
    def batch_analyze(self, headlines: List[str], delay: float = 1.0) -> List[Dict]:
        """
        Analyze multiple headlines with rate limiting
        """
        results = []
        
        for i, headline in enumerate(headlines):
            logger.info(f"Analyzing headline {i+1}/{len(headlines)}")
            result = self.analyze_headline(headline)
            results.append(result)
            
            # Rate limiting
            if i < len(headlines) - 1:
                time.sleep(delay)
        
        return results

class HybridFakeNewsDetector:
    """
    Combines traditional ML models with LLM analysis
    """
    
    def __init__(self, ml_model, llm_analyzer: LLMFakeNewsAnalyzer):
        self.ml_model = ml_model
        self.llm_analyzer = llm_analyzer
    
    def analyze(self, headline: str) -> Dict:
        """
        Combined analysis using both ML model and LLM
        """
        # Get ML prediction
        if hasattr(self.ml_model, 'predict'):
            predictions, probabilities = self.ml_model.predict([headline])
            ml_prediction = "Fake" if predictions[0] == 0 else "Real"
            ml_confidence = max(probabilities[0]) * 100
        else:
            ml_prediction = "Unknown"
            ml_confidence = 0
        
        # Get LLM analysis
        llm_analysis = self.llm_analyzer.analyze_headline(headline)
        
        # Generate explanation
        explanation = self.llm_analyzer.generate_explanation(
            headline, ml_prediction, ml_confidence
        )
        
        return {
            "headline": headline,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "llm_analysis": llm_analysis,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }

def create_fact_checking_database():
    """
    Create a database of common fake news patterns for reference
    """
    fake_news_patterns = [
        "You won't believe what happened next",
        "Doctors hate this one trick",
        "This will change everything",
        "They don't want you to know",
        "Shocking discovery",
        "Miracle cure found",
        "Government covering up",
        "Secret revealed",
        "Amazing breakthrough",
        "This one simple trick"
    ]
    
    return fake_news_patterns

if __name__ == "__main__":
    # Example usage
    analyzer = LLMFakeNewsAnalyzer()
    
    test_headlines = [
        "Scientists discover miracle cure for all diseases",
        "New study shows coffee prevents cancer",
        "Government announces new economic policy"
    ]
    
    for headline in test_headlines:
        print(f"Analyzing: {headline}")
        result = analyzer.analyze_headline(headline)
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 50)
