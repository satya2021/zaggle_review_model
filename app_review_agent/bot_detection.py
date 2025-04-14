import re
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BotDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Common bot/template patterns
        self.template_patterns = [
            r"please (contact|reach|write to) us at",
            r"we (regret|apologize for) the inconvenience",
            r"thank you for (your|the) feedback",
            r"we appreciate your feedback",
            r"please elaborate (on|about) the issue",
            r"we will look into (the|this) matter",
            r"feel free to reach out",
            r"we are sorry to hear",
            r"we will forward this to",
            r"our team will investigate"
        ]
        
        # Genuineness indicators
        self.genuine_patterns = [
            r"specifically",
            r"regarding your point about",
            r"you mentioned that",
            r"i understand your concern about",
            r"based on what you've described",
            r"let me address your",
            r"to clarify the",
            r"in response to your",
            r"i see that you're having trouble with",
            r"you're right about"
        ]

    def analyze_response(self, 
                        response_text: str, 
                        review_text: str,
                        previous_responses: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a response for bot-like patterns and genuineness
        """
        try:
            scores = {
                'bot_probability': self._calculate_bot_probability(response_text),
                'template_score': self._detect_template_patterns(response_text),
                'genuineness_score': self._calculate_genuineness(response_text, review_text),
                'uniqueness_score': self._calculate_uniqueness(response_text, previous_responses or []),
                'context_relevance': self._calculate_context_relevance(response_text, review_text)
            }
            
            # Calculate overall authenticity score
            authenticity_score = (
                (1 - scores['bot_probability']) * 0.3 +
                (1 - scores['template_score']) * 0.2 +
                scores['genuineness_score'] * 0.2 +
                scores['uniqueness_score'] * 0.15 +
                scores['context_relevance'] * 0.15
            )
            
            return {
                'scores': scores,
                'authenticity_score': authenticity_score,
                'is_likely_bot': authenticity_score < 0.5,
                'analysis_timestamp': datetime.now().isoformat(),
                'flags': self._generate_flags(scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response: {str(e)}")
            return {
                'scores': {},
                'authenticity_score': 0.0,
                'is_likely_bot': True,
                'analysis_timestamp': datetime.now().isoformat(),
                'flags': ['error_during_analysis']
            }

    def _calculate_bot_probability(self, text: str) -> float:
        """Calculate probability that response is from a bot"""
        # Check for template patterns
        template_count = sum(1 for pattern in self.template_patterns 
                           if re.search(pattern, text.lower()))
        
        # Check for repetitive phrases
        words = text.lower().split()
        word_freq = Counter(words)
        repetition_score = max(word_freq.values()) / len(words) if words else 0
        
        # Check for uniform sentence lengths
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(lengths) if len(lengths) > 1 else 0
            length_uniformity = 1 / (1 + length_variance)
        else:
            length_uniformity = 1
        
        # Combine factors
        bot_score = (
            (template_count / len(self.template_patterns)) * 0.4 +
            repetition_score * 0.3 +
            length_uniformity * 0.3
        )
        
        return min(max(bot_score, 0.0), 1.0)

    def _detect_template_patterns(self, text: str) -> float:
        """Detect usage of common template phrases"""
        text_lower = text.lower()
        template_matches = sum(1 for pattern in self.template_patterns 
                             if re.search(pattern, text_lower))
        return min(template_matches / len(self.template_patterns), 1.0)

    def _calculate_genuineness(self, response: str, review: str) -> float:
        """Calculate how genuine and personalized the response is"""
        # Check for specific reference to review content
        review_words = set(review.lower().split())
        response_words = set(response.lower().split())
        content_overlap = len(review_words & response_words) / len(review_words) if review_words else 0
        
        # Check for genuine response patterns
        genuine_count = sum(1 for pattern in self.genuine_patterns 
                          if re.search(pattern, response.lower()))
        genuine_score = genuine_count / len(self.genuine_patterns)
        
        # Check for specific details
        has_specific_details = any([
            re.search(r'\d+', response),  # Contains numbers
            re.search(r'(today|tomorrow|yesterday)', response.lower()),  # Time references
            re.search(r'(version|update) \d+\.\d+', response.lower()),  # Version numbers
            re.search(r'(step|steps?) \d+', response.lower())  # Step references
        ])
        
        return min((content_overlap * 0.4 + genuine_score * 0.4 + 
                   (0.2 if has_specific_details else 0)), 1.0)

    def _calculate_uniqueness(self, response: str, previous_responses: List[str]) -> float:
        """Calculate how unique the response is compared to previous responses"""
        if not previous_responses:
            return 1.0
            
        try:
            # Vectorize responses
            all_responses = [response] + previous_responses
            vectors = self.vectorizer.fit_transform(all_responses)
            
            # Calculate similarity with previous responses
            similarities = (vectors[0] * vectors[1:].T).toarray()[0]
            
            # Convert to uniqueness score (inverse of max similarity)
            uniqueness = 1 - (max(similarities) if similarities.size > 0 else 0)
            return max(min(uniqueness, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating uniqueness: {str(e)}")
            return 0.5

    def _calculate_context_relevance(self, response: str, review: str) -> float:
        """Calculate how relevant the response is to the review context"""
        try:
            # Vectorize both texts
            vectors = self.vectorizer.fit_transform([review, response])
            
            # Calculate cosine similarity
            similarity = (vectors[0] * vectors[1].T).toarray()[0][0]
            
            return max(min(similarity, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating context relevance: {str(e)}")
            return 0.5

    def _generate_flags(self, scores: Dict[str, float]) -> List[str]:
        """Generate warning flags based on scores"""
        flags = []
        
        if scores['bot_probability'] > 0.7:
            flags.append('high_bot_probability')
        if scores['template_score'] > 0.7:
            flags.append('excessive_template_usage')
        if scores['genuineness_score'] < 0.3:
            flags.append('low_genuineness')
        if scores['uniqueness_score'] < 0.3:
            flags.append('low_uniqueness')
        if scores['context_relevance'] < 0.3:
            flags.append('low_context_relevance')
            
        return flags 