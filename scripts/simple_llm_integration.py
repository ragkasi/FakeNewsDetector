"""
Simplified LLM Integration for Enhanced Fake News Detection
Fallback version that works around OpenAI client initialization issues
"""

import os
import requests
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLLMFakeNewsAnalyzer:
    """
    Simplified fake news detection using OpenAI API via direct HTTP requests
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. LLM features will be disabled.")
    
    def _make_api_request(self, messages: List[Dict], max_tokens: int = 500) -> Dict:
        """Make direct HTTP request to OpenAI API with rate limiting handling"""
        if not self.api_key:
            return {"error": "No API key available"}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                logger.warning("Rate limit exceeded. Please wait a moment before trying again.")
                return {
                    "error": "Rate limit exceeded. Please wait a moment before trying again.",
                    "rate_limited": True,
                    "retry_after": "Please wait 1-2 minutes before trying again."
                }
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": str(e)}
    
    def analyze_headline(self, headline: str) -> Dict:
        """
        Comprehensive analysis of a news headline using LLM
        """
        if not self.api_key:
            return {"error": "No API key available"}
        
        prompt = f"""
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
        """
        
        messages = [
            {"role": "system", "content": "You are an expert fact-checker and media literacy specialist."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_request(messages)
        
        if "error" in response:
            return response
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse response: {e}")
            return {"analysis": "Analysis completed but response format was unexpected", "error": "Parse error"}
    
    def generate_explanation(self, headline: str, classification: str, confidence: float) -> str:
        """
        Generate human-readable explanation for classification
        """
        if not self.api_key:
            return "LLM explanation not available (no API key)"
        
        prompt = f"""
        Explain why this news headline might be classified as {classification}:
        
        Headline: {headline}
        Classification: {classification}
        Confidence: {confidence}%
        
        Provide a clear, educational explanation that helps users understand:
        1. Key indicators that led to this classification
        2. Common patterns in {classification.lower()} news
        3. How to verify this information independently
        
        Keep the explanation under 150 words and use simple language.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that explains AI classifications in simple terms."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_request(messages, max_tokens=300)
        
        if "error" in response:
            return f"Explanation generation failed: {response['error']}"
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract explanation: {e}")
            return "Explanation generation completed but response format was unexpected"

def test_simple_llm():
    """Test the simplified LLM integration"""
    print("Testing simplified LLM integration...")
    
    # Test with dummy key first
    analyzer = SimpleLLMFakeNewsAnalyzer(api_key="dummy-key")
    result = analyzer.analyze_headline("Test headline")
    print(f"Test result: {result}")
    
    if "error" in result and "No API key" in result["error"]:
        print("Simplified LLM integration is working (needs real API key)")
        return True
    else:
        print("Simplified LLM integration has issues")
        return False

if __name__ == "__main__":
    test_simple_llm()
