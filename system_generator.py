"""
System Message Generator based on Multi-Dimensional Classification

This module analyzes user queries using sentiment, emotion, and intent classification
to dynamically generate appropriate system messages for GPT-OSS.
"""

from datetime import datetime
from transformers import pipeline
from typing import Dict, Optional, Tuple


class SystemMessageGenerator:
    """
    Generates context-aware system messages based on multi-dimensional
    classification of user queries (sentiment, emotion, intent).
    """
    
    def __init__(self, device: int = -1):
        """
        Initialize classifiers.
        
        Args:
            device: Device to run classifiers on. -1 for CPU, 0+ for GPU index.
        """
        self.device = device
        self._sentiment_classifier = None
        self._emotion_classifier = None
        self._intent_classifier = None
        
        # System message templates based on intent
        self.intent_templates = {
            "asking question": "You are a helpful assistant. Provide clear, accurate answers.",
            "seeking emotional support": "You are a compassionate assistant. Be supportive and understanding.",
            "requesting creative content": "You are a creative assistant. Be imaginative and engaging.",
            "asking for code help": "You are a technical expert. Be precise and provide well-commented code.",
            "casual conversation": "You are a friendly assistant. Be conversational and warm.",
            "expressing frustration": "You are a patient assistant. Be calm, empathetic, and solution-focused.",
            "requesting explanation": "You are an educational assistant. Explain concepts clearly and thoroughly.",
        }
        
        # Tone modifiers based on emotion
        self.emotion_tones = {
            "anger": "Be patient, calm, and non-confrontational.",
            "sadness": "Be empathetic, supportive, and encouraging.",
            "joy": "Match the user's positive energy while staying helpful.",
            "fear": "Be reassuring, clear, and confidence-inspiring.",
            "surprise": "Be informative and help clarify.",
            "disgust": "Be understanding and objective.",
            "neutral": "",
        }
        
        # Sentiment adjustments
        self.sentiment_adjustments = {
            "negative": "Be extra patient and encouraging.",
            "positive": "Maintain the positive tone.",
            "neutral": "",
        }
    
    @property
    def sentiment_classifier(self):
        """Lazy load sentiment classifier."""
        if self._sentiment_classifier is None:
            self._sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
        return self._sentiment_classifier
    
    @property
    def emotion_classifier(self):
        """Lazy load emotion classifier."""
        if self._emotion_classifier is None:
            self._emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device
            )
        return self._emotion_classifier
    
    @property
    def intent_classifier(self):
        """Lazy load zero-shot intent classifier."""
        if self._intent_classifier is None:
            self._intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
        return self._intent_classifier
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of the text.
        
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        result = self.sentiment_classifier(text)[0]
        label = result['label'].lower()
        # Map POSITIVE/NEGATIVE to our format
        if label == 'positive':
            return 'positive', result['score']
        elif label == 'negative':
            return 'negative', result['score']
        return 'neutral', result['score']
    
    def analyze_emotion(self, text: str) -> Tuple[str, float]:
        """
        Analyze emotion in the text.
        
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        result = self.emotion_classifier(text)[0]
        return result['label'].lower(), result['score']
    
    def analyze_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the text using zero-shot classification.
        
        Returns:
            Tuple of (intent_label, confidence_score)
        """
        candidate_intents = list(self.intent_templates.keys())
        result = self.intent_classifier(
            text,
            candidate_intents,
            multi_label=False
        )
        return result['labels'][0], result['scores'][0]
    
    def analyze_query(self, user_query: str) -> Dict:
        """
        Perform full multi-dimensional analysis of user query.
        
        Args:
            user_query: The user's input text
            
        Returns:
            Dictionary containing sentiment, emotion, intent analysis
        """
        sentiment, sentiment_score = self.analyze_sentiment(user_query)
        emotion, emotion_score = self.analyze_emotion(user_query)
        intent, intent_score = self.analyze_intent(user_query)
        
        # Determine reasoning effort based on query complexity
        query_lower = user_query.lower()
        word_count = len(user_query.split())
        
        if word_count > 30 or any(word in query_lower for word in ["explain", "analyze", "compare", "evaluate", "why"]):
            reasoning = "high"
        elif any(word in query_lower for word in ["how", "what is", "describe"]) or intent == "asking for code help":
            reasoning = "medium"
        else:
            reasoning = "low"
        
        return {
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'emotion': emotion,
            'emotion_score': emotion_score,
            'intent': intent,
            'intent_score': intent_score,
            'reasoning': reasoning,
        }
    
    def generate(
        self,
        user_query: str,
        include_date: bool = True,
        custom_base: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        Generate a context-aware system message based on query analysis.
        
        Args:
            user_query: The user's input text
            include_date: Whether to include current date in system message
            custom_base: Optional custom base message to use instead of intent-based
            verbose: If True, print analysis details
            
        Returns:
            Generated system message string
        """
        analysis = self.analyze_query(user_query)
        
        if verbose:
            print(f"Analysis: {analysis}")
        
        # Build system message components
        components = []
        
        # Base message (from intent or custom)
        if custom_base:
            components.append(custom_base)
        else:
            base_msg = self.intent_templates.get(
                analysis['intent'],
                "You are a helpful assistant."
            )
            components.append(base_msg)
        
        # Add emotion-based tone if confidence is high enough
        if analysis['emotion_score'] > 0.6:
            tone = self.emotion_tones.get(analysis['emotion'], "")
            if tone:
                components.append(tone)
        
        # Add sentiment adjustment if strong sentiment detected
        if analysis['sentiment_score'] > 0.8 and analysis['sentiment'] != 'neutral':
            adjustment = self.sentiment_adjustments.get(analysis['sentiment'], "")
            if adjustment:
                components.append(adjustment)
        
        # Add date if requested
        if include_date:
            components.append(f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.")
        
        # Add reasoning effort
        components.append(f"Reasoning effort: {analysis['reasoning']}")
        
        return " ".join(components)


class HybridSystemGenerator:
    """
    A faster hybrid approach combining rule-based logic with ML classification.
    Uses only sentiment classifier (lightweight) combined with keyword rules.
    """
    
    def __init__(self, device: int = -1):
        """
        Initialize the hybrid generator.
        
        Args:
            device: Device to run classifier on. -1 for CPU, 0+ for GPU index.
        """
        self.device = device
        self._sentiment_classifier = None
        
        # Intent keywords for rule-based detection
        self.intent_keywords = {
            "creative": ["story", "tale", "once upon", "poem", "write me", "creative"],
            "technical": ["code", "debug", "function", "error", "program", "script", "api"],
            "emotional": ["sad", "frustrated", "upset", "angry", "worried", "scared", "anxious", "depressed"],
            "math": ["calculate", "solve", "equation", "math", "sum", "multiply", "divide"],
            "explanation": ["explain", "why", "how does", "what is", "describe", "tell me about"],
        }
        
        # Intent classification instruction appended to every system prompt
        self.intent_instruction = (
            "Before your final response, classify the user's intent into exactly one of: "
            "question, creative, technical, emotional, math, explanation, translate, search, none. "
            "Output it as: <|channel|>intent<|message|>LABEL "
            "Then output your response as: <|channel|>final<|message|>YOUR RESPONSE"
        )

        # Templates for each intent
        self.templates = {
            "question": "You are a helpful assistant.",
            "creative": "You are a creative storyteller. Be imaginative and engaging.",
            "technical": "You are a technical expert. Be precise and provide clear solutions.",
            "emotional": "You are a compassionate assistant. Be supportive and understanding.",
            "math": "You are a mathematics tutor. Show your work step-by-step.",
            "explanation": "You are an educational assistant. Explain concepts clearly.",
        }
    
    @property
    def sentiment_classifier(self):
        """Lazy load sentiment classifier."""
        if self._sentiment_classifier is None:
            self._sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
        return self._sentiment_classifier
    
    def detect_intent_by_rules(self, user_query: str) -> str:
        """
        Detect intent using keyword matching rules.
        
        Args:
            user_query: The user's input text
            
        Returns:
            Detected intent string
        """
        query_lower = user_query.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "question"
    
    def analyze(self, user_query: str) -> Dict:
        """
        Perform hybrid analysis (rules + sentiment ML).
        
        Args:
            user_query: The user's input text
            
        Returns:
            Analysis dictionary
        """
        query_lower = user_query.lower()
        
        # Rule-based intent
        intent = self.detect_intent_by_rules(user_query)
        
        # ML-based sentiment
        sentiment_result = self.sentiment_classifier(user_query)[0]
        sentiment = sentiment_result['label'].lower()
        sentiment_score = sentiment_result['score']
        
        # Determine reasoning level
        word_count = len(user_query.split())
        if word_count > 30 or "explain" in query_lower or "why" in query_lower:
            reasoning = "high"
        elif "how" in query_lower or intent == "technical":
            reasoning = "medium"
        else:
            reasoning = "low"
        
        return {
            'intent': intent,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'reasoning': reasoning,
        }
    
    def generate(
        self,
        user_query: str,
        include_date: bool = True,
        verbose: bool = False
    ) -> str:
        """
        Generate system message using hybrid analysis.
        
        Args:
            user_query: The user's input text
            include_date: Whether to include current date
            verbose: If True, print analysis details
            
        Returns:
            Generated system message string
        """
        analysis = self.analyze(user_query)
        
        if verbose:
            print(f"Analysis: {analysis}")
        
        # Build components
        components = []
        
        # Base message from intent
        base_msg = self.templates.get(analysis['intent'], self.templates['question'])
        components.append(base_msg)
        
        # Sentiment-based tone adjustment
        if analysis['sentiment'] == 'negative' and analysis['sentiment_score'] > 0.7:
            components.append("Be patient and encouraging.")
        elif analysis['sentiment'] == 'positive' and analysis['sentiment_score'] > 0.7:
            components.append("Match their enthusiasm.")
        
        # Date
        if include_date:
            components.append(f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.")
        
        # Reasoning effort
        components.append(f"Reasoning effort: {analysis['reasoning']}")
        
        # Intent classification instruction
        components.append(self.intent_instruction)
        
        return " ".join(components)


class SimpleSystemGenerator:
    """
    A minimal rule-based system message generator.
    No ML dependencies - fastest option.
    """
    
    def __init__(self):
        self.intent_keywords = {
            "creative": ["story", "tale", "once upon", "poem", "creative", "imagine"],
            "technical": ["code", "debug", "function", "error", "program", "script"],
            "emotional": ["sad", "frustrated", "upset", "angry", "worried", "help me"],
            "math": ["calculate", "solve", "equation", "math", "sum"],
        }
        
        self.templates = {
            "default": "You are a helpful assistant.",
            "creative": "You are a creative storyteller.",
            "technical": "You are a technical expert.",
            "emotional": "You are a compassionate assistant. Be supportive.",
            "math": "You are a mathematics tutor.",
        }
    
    def generate(
        self,
        user_query: str,
        reasoning: str = "low",
        include_date: bool = True
    ) -> str:
        """
        Generate system message using simple rules.
        
        Args:
            user_query: The user's input text
            reasoning: Reasoning effort level ("low", "medium", "high")
            include_date: Whether to include current date
            
        Returns:
            Generated system message string
        """
        query_lower = user_query.lower()
        
        # Detect intent
        detected_intent = "default"
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intent = intent
                break
        
        # Build message
        components = [self.templates[detected_intent]]
        
        if include_date:
            components.append(f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.")
        
        components.append(f"Reasoning effort: {reasoning}")
        
        return " ".join(components)


def format_prompt_with_system(
    user_query: str,
    system_message: str,
    tokenizer=None
) -> str:
    """
    Format a complete prompt with system message in Harmony format.
    
    Args:
        user_query: The user's question/input
        system_message: Generated system message
        tokenizer: Optional tokenizer to encode the prompt
        
    Returns:
        Formatted prompt string (or token list if tokenizer provided)
    """
    prompt = (
        f"<|start|>system<|message|>{system_message}<|end|>"
        f"<|start|>user<|message|>{user_query}<|end|>"
        f"<|start|>assistant"
    )
    
    if tokenizer is not None:
        return tokenizer.encode(prompt, allowed_special='all')
    
    return prompt


# Example usage
if __name__ == "__main__":
    # Test the generators
    test_queries = [
        "What is the capital of France?",
        "I'm so frustrated with this bug, nothing works!",
        "Tell me a story about a dragon",
        "Explain how neural networks learn",
        "Write a Python function to reverse a string",
    ]
    
    print("=" * 60)
    print("Testing SimpleSystemGenerator (No ML)")
    print("=" * 60)
    simple_gen = SimpleSystemGenerator()
    for query in test_queries:
        system_msg = simple_gen.generate(query)
        print(f"\nQuery: {query}")
        print(f"System: {system_msg}")
    
    print("\n" + "=" * 60)
    print("Testing HybridSystemGenerator (Lightweight ML)")
    print("=" * 60)
    hybrid_gen = HybridSystemGenerator(device=-1)
    for query in test_queries:
        system_msg = hybrid_gen.generate(query, verbose=True)
        print(f"Query: {query}")
        print(f"System: {system_msg}\n")
    
    print("\n" + "=" * 60)
    print("Testing SystemMessageGenerator (Full ML)")
    print("=" * 60)
    full_gen = SystemMessageGenerator(device=-1)
    for query in test_queries:
        system_msg = full_gen.generate(query, verbose=True)
        print(f"Query: {query}")
        print(f"System: {system_msg}\n")
