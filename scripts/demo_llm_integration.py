"""
Demo LLM Integration for Testing Without API Calls
Provides mock responses for demonstration purposes
"""

import json
import random
from typing import Dict, List, Optional
from datetime import datetime

class DemoLLMFakeNewsAnalyzer:
    """
    Demo fake news detection using mock LLM responses
    Useful for testing and demonstration without API costs
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = api_key
        print("Demo LLM Analyzer initialized (no API calls will be made)")
    
    def analyze_headline(self, headline: str) -> Dict:
        """
        Mock analysis of a news headline using predefined responses
        """
        # Simple keyword-based mock analysis
        headline_lower = headline.lower()
        
        # Mock red flags based on common fake news patterns
        red_flags = []
        if any(word in headline_lower for word in ['miracle', 'cure', 'breakthrough', 'shocking', 'amazing']):
            red_flags.append("Uses sensational language typical of clickbait")
        if any(word in headline_lower for word in ['scientists', 'study', 'research']) and len(headline.split()) < 10:
            red_flags.append("Claims scientific backing without sufficient detail")
        if any(word in headline_lower for word in ['government', 'official', 'announces']):
            red_flags.append("Makes official-sounding claims that may be unverified")
        
        # Mock confidence based on red flags
        confidence = "High" if len(red_flags) >= 2 else "Medium" if len(red_flags) == 1 else "Low"
        
        # Mock analysis text
        if len(red_flags) >= 2:
            analysis = "This headline contains multiple red flags typical of misleading or sensationalized content. The language and structure suggest it may be designed to generate clicks rather than inform."
        elif len(red_flags) == 1:
            analysis = "This headline shows some characteristics that warrant further verification. While not definitively fake, it contains elements that should be fact-checked."
        else:
            analysis = "This headline appears to be straightforward and doesn't contain obvious red flags. However, all news should be verified through reliable sources."
        
        # Mock verification steps
        verification_steps = [
            "Check the source's credibility and history",
            "Look for the same story on established news outlets",
            "Verify any specific claims or statistics mentioned",
            "Check if the story has been fact-checked by reputable organizations"
        ]
        
        return {
            "analysis": analysis,
            "red_flags": red_flags,
            "confidence": confidence,
            "verification_steps": verification_steps,
            "demo_mode": True
        }
    
    def generate_explanation(self, headline: str, classification: str, confidence: float) -> str:
        """
        Generate mock explanation for classification
        """
        explanations = {
            "fake": f"This headline was classified as potentially fake news with {confidence}% confidence. Common indicators include sensational language, lack of credible sources, or claims that seem too good to be true.",
            "real": f"This headline was classified as likely real news with {confidence}% confidence. It appears to use standard journalistic language and doesn't contain obvious red flags.",
            "unreliable": f"This headline was classified as unreliable with {confidence}% confidence. It may contain some factual elements but also includes misleading or unverified claims."
        }
        
        return explanations.get(classification.lower(), 
                              f"This headline was classified as {classification} with {confidence}% confidence. The AI model detected patterns that suggest this classification.")

def test_demo_llm():
    """Test the demo LLM integration"""
    print("Testing demo LLM integration...")
    
    analyzer = DemoLLMFakeNewsAnalyzer()
    
    test_headlines = [
        "Scientists discover miracle cure for all diseases",
        "Government announces new economic policy",
        "Breaking: Major breakthrough in renewable energy"
    ]
    
    for headline in test_headlines:
        print(f"\nTesting: {headline}")
        result = analyzer.analyze_headline(headline)
        print(f"Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_demo_llm()
