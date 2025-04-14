# tasks.py
import logging
import json
from typing import Dict, Any, List, Union, Optional, Tuple
from functools import lru_cache
import os
from datetime import datetime
from pathlib import Path
import random
from celery import Celery
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from openai import OpenAI
from response_db import ResponseDatabase
from bot_detection import BotDetector
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Celery
celery = Celery('tasks')
celery.conf.update(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global variables for FAQ handling
FAQ_FILE = 'data/faq.csv'
faq_data = None
faq_vectorizer = None
faq_vectors = None

# Default FAQ entries to prevent empty vocabulary
DEFAULT_FAQS = [
    {
        'question': 'How do I activate my card?',
        'answer': 'To activate your card, please call the number on the back of your card or visit our secure website. The activation process typically takes just a few minutes.'
    },
    {
        'question': 'What should I do if I experience delays in card activation?',
        'answer': 'If you experience delays in card activation, please contact our support team at 1-800-XXX-XXXX. We aim to resolve activation issues within 2 hours.'
    },
    {
        'question': 'How can I contact customer support?',
        'answer': 'You can reach our customer support team 24/7 through multiple channels: Phone: 1-800-XXX-XXXX, Email: support@example.com, or Live Chat on our website.'
    }
]

# Initialize response database
response_db = ResponseDatabase()

# Initialize bot detector
bot_detector = BotDetector()

def initialize_faq_index() -> bool:
    """
    Initialize the FAQ index by creating the necessary directory and file structure.
    If the FAQ file doesn't exist or is empty, populate it with default FAQs.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # Create data directory if it doesn't exist
        Path('data').mkdir(exist_ok=True)
        
        # Check if FAQ file exists and has content
        faq_path = Path(FAQ_FILE)
        if not faq_path.exists() or faq_path.stat().st_size == 0:
            # Create new FAQ file with default content
            df = pd.DataFrame(DEFAULT_FAQS)
            # Ensure directory exists
            faq_path.parent.mkdir(parents=True, exist_ok=True)
            # Save with required columns
            df.to_csv(FAQ_FILE, index=False)
            logging.info(f"Created new FAQ file with {len(DEFAULT_FAQS)} default entries")
            return True
            
        # If file exists, validate and update if necessary
        df = pd.read_csv(FAQ_FILE)
        
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Check for required columns
        required_columns = {'question', 'answer'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            # If missing required columns, create new file with default content
            df = pd.DataFrame(DEFAULT_FAQS)
            df.to_csv(FAQ_FILE, index=False)
            logging.warning(f"Recreated FAQ file due to missing columns: {missing_columns}")
            return True
            
        # Ensure there's at least one valid entry
        if len(df) == 0:
            df = pd.DataFrame(DEFAULT_FAQS)
            df.to_csv(FAQ_FILE, index=False)
            logging.warning("Populated empty FAQ file with default entries")
            return True
            
        # Clean the data
        df['question'] = df['question'].fillna('')
        df['answer'] = df['answer'].fillna('')
        df = df[df['question'].str.strip().str.len() > 0]
        
        if len(df) == 0:
            # If no valid entries after cleaning, add defaults
            df = pd.DataFrame(DEFAULT_FAQS)
            logging.warning("Added default FAQs after cleaning resulted in empty dataset")
        
        # Save cleaned data
        df.to_csv(FAQ_FILE, index=False)
        logging.info(f"Successfully initialized FAQ index with {len(df)} entries")
        return True
        
    except Exception as e:
        logging.error(f"Error initializing FAQ index: {str(e)}")
        # Create with defaults on error
        try:
            df = pd.DataFrame(DEFAULT_FAQS)
            Path('data').mkdir(exist_ok=True)
            df.to_csv(FAQ_FILE, index=False)
            logging.warning("Created FAQ file with defaults after error")
            return True
        except Exception as e2:
            logging.error(f"Critical error creating FAQ file: {str(e2)}")
            return False

def load_faq_index(force_reload: bool = False) -> bool:
    """Load and initialize the FAQ index"""
    global faq_data, faq_vectorizer, faq_vectors
    
    try:
        if faq_data is not None and not force_reload:
            return True
        
        # Initialize with default FAQs if needed
        initialize_faq_index()
        
        # Load FAQ data
        faq_data = pd.read_csv(FAQ_FILE)
        
        # Normalize column names
        faq_data.columns = [col.lower() for col in faq_data.columns]
        
        # Ensure required columns exist
        required_columns = {'question', 'answer'}
        if not all(col in faq_data.columns for col in required_columns):
            missing_cols = required_columns - set(faq_data.columns)
            raise ValueError(f"Missing required columns in FAQ file: {missing_cols}")
        
        # Ensure we have at least one valid question
        if len(faq_data) == 0 or faq_data['question'].isna().all():
            faq_data = pd.DataFrame(DEFAULT_FAQS)
        
        # Clean the questions
        faq_data['question'] = faq_data['question'].fillna('')
        faq_data['question'] = faq_data['question'].astype(str).apply(lambda x: x.strip())
        
        # Remove empty questions
        faq_data = faq_data[faq_data['question'].str.len() > 0]
        
        # Initialize TF-IDF vectorizer with minimal parameters
        faq_vectorizer = TfidfVectorizer(
            stop_words=None,  # Don't remove stop words
            max_features=None,  # Don't limit features
            ngram_range=(1, 1),  # Use only unigrams
            min_df=1,  # Include all terms
            token_pattern=r'(?u)\b\w+\b'  # Match any word character
        )
        
        # Create TF-IDF vectors for FAQ questions
        faq_vectors = faq_vectorizer.fit_transform(faq_data['question'])
        
        logger.info(f"Successfully loaded FAQ index with {len(faq_data)} entries")
        return True
        
    except Exception as e:
        logger.error(f"Error loading FAQ index: {str(e)}")
        # Initialize with default values on error
        faq_data = pd.DataFrame(DEFAULT_FAQS)
        faq_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        faq_vectors = faq_vectorizer.fit_transform(faq_data['question'])
        return False

def get_relevant_faqs(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Get relevant FAQ entries for a given query"""
    try:
        if faq_data is None or faq_vectorizer is None or faq_vectors is None:
            load_faq_index()
        
        # Ensure query is not empty
        if not query or not query.strip():
            return []
        
        # Transform query using the same vectorizer
        query_vector = faq_vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, faq_vectors).flatten()
        
        # Get top-k most similar FAQs
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Filter by minimum similarity threshold
        threshold = 0.1
        relevant_faqs = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                relevant_faqs.append({
                    'question': faq_data.iloc[idx]['question'],
                    'answer': faq_data.iloc[idx]['answer'],
                    'similarity': float(similarities[idx])
                })
        
        return relevant_faqs
        
    except Exception as e:
        logger.error(f"Error getting relevant FAQs: {str(e)}")
        return []

def format_faq_context(faqs: List[Dict[str, str]]) -> str:
    """Format FAQ entries into a context string"""
    try:
        if not faqs:
            return ""
        
        context_parts = []
        for faq in faqs:
            q = faq.get('question', '').strip()
            a = faq.get('answer', '').strip()
            if q and a:
                context_parts.append(f"Q: {q}\nA: {a}")
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error formatting FAQ context: {str(e)}")
        return ""

# Initialize FAQ index on module load
load_faq_index()

class SentimentAnalyzer:
    """Class to handle sentiment analysis using both rule-based and OpenAI approaches"""
    
    def __init__(self):
        # Performance and technical issues
        self.performance_keywords = [
            'slow', 'lag', 'loading', 'load time', 'performance', 'speed',
            'freeze', 'crash', 'unresponsive', 'delay', 'waiting',
            'frustrating', 'annoying', 'terrible', 'awful', 'horrible',
            'bug', 'error', 'issue', 'problem', 'not working'
        ]
        
        self.performance_phrases = [
            'takes forever', 'too slow', 'very slow', 'extremely slow',
            'unbearably slow', 'painfully slow', 'ridiculously slow',
            'losing patience', 'frustrated with', 'annoyed by',
            'keeps crashing', 'always crashes', 'frequently crashes',
            'constant issues', 'regular problems'
        ]
        
        # User experience and satisfaction
        self.positive_keywords = [
            'fast', 'quick', 'responsive', 'smooth', 'efficient',
            'excellent', 'great', 'good', 'amazing', 'wonderful',
            'love', 'perfect', 'best', 'outstanding', 'impressive',
            'easy', 'simple', 'intuitive', 'user-friendly', 'helpful'
        ]
        
        self.negative_keywords = [
            'bad', 'poor', 'terrible', 'awful', 'horrible',
            'frustrating', 'annoying', 'disappointing', 'unacceptable',
            'worst', 'hate', 'dislike', 'problem', 'issue',
            'difficult', 'complicated', 'confusing', 'hard to use'
        ]
        
        # Emotional context
        self.emotional_keywords = {
            'frustration': ['frustrated', 'annoyed', 'angry', 'upset', 'disappointed'],
            'satisfaction': ['happy', 'pleased', 'satisfied', 'delighted', 'impressed'],
            'confusion': ['confused', 'lost', 'uncertain', 'unsure', 'puzzled'],
            'urgency': ['urgent', 'critical', 'important', 'emergency', 'immediate']
        }
        
        # Feature-specific feedback
        self.feature_keywords = {
            'ui': ['interface', 'design', 'layout', 'look', 'appearance'],
            'functionality': ['feature', 'function', 'capability', 'ability', 'option'],
            'support': ['help', 'support', 'assistance', 'customer service', 'service'],
            'performance': ['speed', 'performance', 'loading', 'response time', 'lag']
        }
        
        self.intensifiers = [
            'very', 'extremely', 'absolutely', 'completely', 'totally',
            'really', 'so', 'too', 'incredibly', 'unbelievably'
        ]
        
        self.negations = [
            'not', 'no', 'never', 'none', 'neither', 'nor',
            'doesn\'t', 'don\'t', 'won\'t', 'can\'t', 'couldn\'t'
        ]
        
        # Load OpenAI config
        self.load_openai_config()

    def load_openai_config(self) -> None:
        """Load OpenAI configuration from file"""
        try:
            config_file = Path('config/openai_sentiment_config.json')
            if config_file.exists():
                with open(config_file) as f:
                    self.openai_config = json.load(f)
            else:
                self.openai_config = {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.3,
                    'max_tokens': 150,
                    'system_prompt': 'You are a sentiment analysis expert. Analyze the review and provide accurate sentiment analysis.',
                    'response_format': {
                        'label': 'sentiment_label',
                        'score': 'confidence_score',
                        'issues': {
                            'performance': 'boolean',
                            'support': 'boolean',
                            'features': 'boolean',
                            'other': 'boolean'
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Error loading OpenAI config: {str(e)}")
            self.openai_config = {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.3,
                'max_tokens': 150,
                'system_prompt': 'You are a sentiment analysis expert. Analyze the review and provide accurate sentiment analysis.',
                'response_format': {
                    'label': 'sentiment_label',
                    'score': 'confidence_score',
                    'issues': {
                        'performance': 'boolean',
                        'support': 'boolean',
                        'features': 'boolean',
                        'other': 'boolean'
                    }
                }
            }

    def verify_with_openai(self, text: str, initial_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Verify sentiment with OpenAI"""
        try:
            prompt = f"""Analyze the sentiment of this review and verify if it's correct:
            Review: "{text}"
            Initial Analysis: {initial_sentiment}
            
            Please provide:
            1. Final sentiment label (very_positive, positive, neutral, negative, very_negative)
            2. Confidence score (0.0 to 1.0)
            3. Key issues identified (if any)
            
            Format your response as JSON with these fields:
            {{
                "label": "sentiment_label",
                "score": confidence_score,
                "issues": {{
                    "performance": boolean,
                    "support": boolean,
                    "features": boolean,
                    "other": boolean
                }}
            }}"""

            response = openai_client.chat.completions.create(
                model=self.openai_config['model'],
                messages=[
                    {"role": "system", "content": self.openai_config['system_prompt']},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.openai_config['temperature'],
                max_tokens=self.openai_config['max_tokens']
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # If OpenAI detects performance issues, ensure it's marked as negative
            if result.get('issues', {}).get('performance', False):
                result['label'] = 'negative'
                result['score'] = min(result.get('score', 0.5), 0.3)
            
            return result

        except Exception as e:
            logger.error(f"Error verifying sentiment with OpenAI: {str(e)}")
            return initial_sentiment

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a review text"""
        try:
            # Convert to lowercase for case-insensitive matching
            text_lower = text.lower()
            
            # Initialize sentiment analysis result
            sentiment_result = {
                'label': 'neutral',
                'score': 0.5,
                'issues': {},
                'context': {
                    'emotions': [],
                    'features': [],
                    'urgency': False
                }
            }
            
            # Check for performance issues
            has_performance_issue = any(keyword in text_lower for keyword in self.performance_keywords)
            has_performance_phrase = any(phrase in text_lower for phrase in self.performance_phrases)
            
            # Analyze emotional context
            for emotion, keywords in self.emotional_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    sentiment_result['context']['emotions'].append(emotion)
            
            # Analyze feature-specific feedback
            for feature, keywords in self.feature_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    sentiment_result['context']['features'].append(feature)
            
            # Check for urgency
            sentiment_result['context']['urgency'] = any(
                keyword in text_lower for keyword in self.emotional_keywords['urgency']
            )
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
            negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
            
            # Check for intensifiers and negations
            has_intensifier = any(word in text_lower for word in self.intensifiers)
            has_negation = any(word in text_lower for word in self.negations)
            
            # Calculate base sentiment score
            if positive_count > negative_count:
                base_score = 0.7
            elif negative_count > positive_count:
                base_score = 0.3
            else:
                base_score = 0.5
            
            # Adjust score based on intensifiers and negations
            if has_intensifier:
                if base_score > 0.5:
                    base_score = min(1.0, base_score + 0.2)
                else:
                    base_score = max(0.0, base_score - 0.2)
            
            if has_negation:
                base_score = 1.0 - base_score  # Invert the sentiment
            
            # If performance issue is mentioned, consider it negative
            if has_performance_issue or has_performance_phrase:
                base_score = min(base_score, 0.3)
                sentiment_result['issues']['performance'] = True
            
            # Determine sentiment label
            if base_score >= 0.8:
                sentiment_result['label'] = 'very_positive'
            elif base_score >= 0.6:
                sentiment_result['label'] = 'positive'
            elif base_score >= 0.4:
                sentiment_result['label'] = 'neutral'
            elif base_score >= 0.2:
                sentiment_result['label'] = 'negative'
            else:
                sentiment_result['label'] = 'very_negative'
            
            sentiment_result['score'] = base_score
            
            # Verify with OpenAI
            return self.verify_with_openai(text, sentiment_result)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'label': 'neutral',
                'score': 0.5,
                'issues': {},
                'context': {
                    'emotions': [],
                    'features': [],
                    'urgency': False
                }
            }

# Create global instance of SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer()

def generate_response(
    text: str,
    sentiment_analysis: Dict[str, Any],
    settings: Dict[str, Any]
) -> str:
    """Generate response based on analysis and past successful responses"""
    try:
        # Find similar past responses
        similar_responses = response_db.find_similar_responses(text)
        
        # If we have good similar responses, use them as examples
        response_examples = ""
        if similar_responses:
            response_examples = "\n\nHere are some successful past responses to similar reviews:\n" + \
                "\n".join([f"Example {i+1}: {r['response']}" for i, r in enumerate(similar_responses)])
        
        # Build the prompt
        prompt_parts = [
            {
                "role": "system",
                "content": f"""You are a highly skilled customer service representative responding to app reviews. 
                Your task is to generate a helpful, empathetic response addressing the user's concerns.

                Response Style Guidelines:
                - Empathy Level: {settings.get('empathy_level', 3)}/5 (higher means more empathetic)
                - Creativity Level: {settings.get('creativity_level', 3)}/5 (higher means more creative language)
                - {'' if settings.get('include_personal_experience') else 'Do not '}include personal experiences
                - {'' if settings.get('include_examples') else 'Do not '}include examples
                - {'' if settings.get('include_metaphors') else 'Do not '}use metaphors
                
                If relevant, use this FAQ information:
                {format_faq_context(get_relevant_faqs(text))}

                {response_examples}

                Response Guidelines:
                1. Start with an appropriate greeting and acknowledgment
                2. Address specific issues mentioned in the review
                3. Provide clear, actionable solutions
                4. Match your tone to the sentiment of the review
                5. Be concise but thorough
                6. End with a positive, forward-looking statement
                """
            },
            {
                "role": "user",
                "content": f"""Review Text: {text}

                Sentiment Analysis:
                - Overall Sentiment: {sentiment_analysis.get('label', 'neutral')}
                - Sentiment Score: {sentiment_analysis.get('score', 0.5)}
                - Detected Issues: {', '.join(k for k, v in sentiment_analysis.get('issues', {}).items() if v)}

                Please generate an appropriate response to this review, taking inspiration from the successful past responses if provided."""
            }
        ]
        
        # Generate response using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt_parts,
            temperature=min(0.3 + (settings.get('creativity_level', 3) * 0.1), 0.9),
            max_tokens=500,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        # Update use count for similar responses that were used
        if similar_responses:
            for r in similar_responses:
                r['use_count'] = r.get('use_count', 0) + 1
            
        return generated_text
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"""I apologize, but I encountered an error while generating a response. 
        
Our team has been notified and is working to resolve this issue. In the meantime, please:
1. Try submitting your review again
2. Contact our support team directly at support@example.com
3. Visit our help center at help.example.com

Error details: {str(e)}"""

@celery.task
def process_review_task(
    review_text: str,
    empathy_level: int = 3,
    creativity_level: int = 3,
    include_personal_experience: bool = False,
    include_examples: bool = False,
    include_metaphors: bool = False
) -> Dict[str, Union[str, Dict]]:
    """Process a review and generate a response."""
    try:
        # Generate a unique review ID
        review_id = str(uuid.uuid4())
        
        # Initialize sentiment analyzer if needed
        sentiment_analyzer = SentimentAnalyzer()
        
        # Analyze sentiment
        sentiment_analysis = sentiment_analyzer.analyze_sentiment(review_text)
        
        # Generate response first
        response = generate_response(
            text=review_text,
            sentiment_analysis=sentiment_analysis,
            settings={
                'empathy_level': empathy_level,
                'creativity_level': creativity_level,
                'include_personal_experience': include_personal_experience,
                'include_examples': include_examples,
                'include_metaphors': include_metaphors
            }
        )
        
        # Now analyze the generated response for bot-like patterns
        bot_analysis = bot_detector.analyze_response(
            response_text=response,  # Analyze the generated response
            review_text=review_text,
            previous_responses=[]  # Optionally get previous responses from the database
        )
        
        logger.info(f"Bot analysis results: {bot_analysis}")
        
        # Convert boolean values to integers for JSON serialization
        bot_analysis['is_likely_bot'] = 1 if bot_analysis.get('is_likely_bot', True) else 0
        
        # Ensure scores are properly formatted
        if 'scores' in bot_analysis:
            bot_analysis['scores'] = {
                k: float(v) for k, v in bot_analysis['scores'].items()
            }
        
        # Save response with feedback data
        feedback_data = {
            'empathy_score': empathy_level,
            'relevance_score': 3,  # Default value
            'helpfulness_score': 3,  # Default value
            'is_good': True,  # Assume good until feedback received
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to response database
        response_db.save_response(
            review_text=review_text,
            response=response,
            feedback_data=feedback_data
        )
        
        result = {
            'review_id': review_id,
            'response': response,
            'sentiment': sentiment_analysis,
            'bot_analysis': bot_analysis,
            'success': 1,
            'error': None
        }
        
        logger.info(f"Task result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        return {
            'review_id': None,
            'response': None,
            'sentiment': None,
            'bot_analysis': None,
            'success': 0,
            'error': str(e)
        }

def calculate_bot_detection_metrics(review_text: str) -> Dict[str, Any]:
    """
    Calculate metrics to detect potential bot-generated reviews.

    Args:
        review_text (str): The review text to analyze

    Returns:
        Dict[str, Any]: Dictionary containing bot detection metrics
    """
    if not review_text:
        return {
            'is_likely_bot': False,
            'confidence': 0.0,
            'metrics': {},
            'flags': []
        }
    
    # Initialize metrics
    metrics = {}
    flags = []
    
    # 1. Text length analysis
    metrics['text_length'] = len(review_text)
    if metrics['text_length'] < 10:
        flags.append('very_short_text')
    elif metrics['text_length'] > 2000:
        flags.append('unusually_long_text')
    
    # 2. Repetition analysis
    words = re.findall(r'\b\w+\b', review_text.lower())
    word_counts = Counter(words)
    
    # Calculate word repetition ratio
    unique_words = len(word_counts)
    total_words = len(words)
    metrics['unique_word_ratio'] = unique_words / total_words if total_words > 0 else 0
    
    if metrics['unique_word_ratio'] < 0.4:
        flags.append('high_word_repetition')
    
    # 3. Pattern detection
    # Check for repeated punctuation
    if re.search(r'([!?.])\1{2,}', review_text):
        flags.append('repeated_punctuation')
    
    # Check for excessive capitalization
    caps_ratio = sum(1 for c in review_text if c.isupper()) / len(review_text) if review_text else 0
    metrics['caps_ratio'] = caps_ratio
    if caps_ratio > 0.5:
        flags.append('excessive_caps')
    
    # 4. Spam patterns
    spam_patterns = [
        r'\b(buy|sell|discount|offer|price|deal|order|purchase)\b.*\b(now|today|limited|exclusive)\b',
        r'https?://\S+',
        r'\b\d+%\s*(off|discount)\b',
        r'\b(click|visit|check|see)\b.*\b(link|site|page|website)\b'
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, review_text.lower()):
            flags.append('promotional_content')
            break
    
    # 5. Time pattern analysis
    # Check for timestamp-like patterns that bots might leave
    if re.search(r'\b\d{2}:\d{2}(:\d{2})?\b', review_text):
        flags.append('contains_timestamp')
    
    # Calculate overall bot likelihood
    num_flags = len(flags)
    confidence = min(0.1 * num_flags, 1.0)  # 10% per flag, max 100%
    
    # Determine if likely bot based on flags and metrics
    is_likely_bot = (
        num_flags >= 3 or  # Multiple suspicious patterns
        metrics['unique_word_ratio'] < 0.3 or  # Extremely repetitive
        ('promotional_content' in flags and num_flags >= 2)  # Promotional content with other flags
    )
    
    return {
        'is_likely_bot': is_likely_bot,
        'confidence': confidence,
        'metrics': metrics,
        'flags': flags
    }

@celery.task(name='tasks.process_feedback')
def process_feedback(review_id: str, response: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process feedback for a generated response.
    
    Args:
        review_id (str): Unique identifier for the review
        response (str): The generated response that received feedback
        feedback_data (Dict[str, Any]): Dictionary containing feedback metrics and comments
        
    Returns:
        Dict[str, Any]: Processing result
    """
    try:
        # Log the feedback
        logger.info(f"Processing feedback for review {review_id}")
        logger.info(f"Feedback data: {feedback_data}")
        
        # Extract feedback metrics
        empathy_score = feedback_data.get('empathy_score', 3)
        relevance_score = feedback_data.get('relevance_score', 3)
        helpfulness_score = feedback_data.get('helpfulness_score', 3)
        is_good = feedback_data.get('is_good', False)
        comments = feedback_data.get('comments', '')
        
        # Store feedback in a CSV file
        feedback_file = Path('data/feedback.csv')
        feedback_file.parent.mkdir(exist_ok=True)
        
        # Prepare feedback entry
        feedback_entry = {
            'review_id': review_id,
            'timestamp': datetime.now().isoformat(),
            'response': response,
            'empathy_score': empathy_score,
            'relevance_score': relevance_score,
            'helpfulness_score': helpfulness_score,
            'is_good': is_good,
            'comments': comments
        }
        
        # Append to CSV
        df = pd.DataFrame([feedback_entry])
        if not feedback_file.exists():
            df.to_csv(feedback_file, index=False)
        else:
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        
        return {
            'status': 'success',
            'message': 'Feedback processed successfully',
            'review_id': review_id
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return {
            'status': 'error',
            'message': f'Error processing feedback: {str(e)}',
            'review_id': review_id
        }

if __name__ == '__main__':
    celery.start()
