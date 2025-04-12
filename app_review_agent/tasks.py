# tasks.py
import os
from celery import Celery
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import logging
from llama_cpp import Llama
import json
from pathlib import Path
from fuzzywuzzy import fuzz
from functools import lru_cache
import concurrent.futures
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
from celery.signals import worker_init
from datetime import datetime
import torch
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai


load_dotenv()

# Initialize Celery
celery = Celery(__name__, 
                broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
                backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))
celery.config_from_object('celery_config')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faq.index')

openai_client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
sentiment_analyzer = SentimentIntensityAnalyzer()

logger = logging.getLogger(__name__)

# Configuration constants
CONFIG_FILE = Path('config.json')
DEFAULT_CONFIG = {
    'llm_provider': 'openai',
    'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
    'llama_model_path': ''
}

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # If config doesn't exist, use default config with environment variables
        return DEFAULT_CONFIG


# Global variables
faq_data = None
question_list = None
encoder = None
faiss_index = None

@worker_init.connect
def init_worker(**kwargs):
    """Initialize resources when worker starts"""
    global faq_data, question_list, encoder, faiss_index
    initialize_faq_index()

def initialize_faq_index() -> bool:
    """Initialize the FAQ index from CSV file with FAISS"""
    global faq_data, question_list, encoder, faiss_index
    try:
        # Load FAQ data from CSV
        csv_path = os.path.join('data', 'faq.csv')
        if not os.path.exists(csv_path):
            logger.warning(f"FAQ file not found at {csv_path}")
            return False
            
        faq_data = pd.read_csv(csv_path)
        logger.info(f"Loaded FAQ data with shape: {faq_data.shape}")
        
        # Store questions separately
        question_list = faq_data['User Query'].tolist()
        if not question_list:
            logger.warning("No questions found in FAQ data")
            return False
            
        logger.info(f"Number of questions loaded: {len(question_list)}")
        
        # Initialize sentence transformer
        encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        
        # Create embeddings
        embeddings = encoder.encode(question_list, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy().astype('float32')
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
        
        logger.info(f"FAISS index created with {faiss_index.ntotal} vectors")
        return True
    except Exception as e:
        logger.error(f"Error initializing FAQ index: {str(e)}")
        return False

def load_faq_index():
    """Load the FAISS index if it exists, otherwise create it"""
    global faq_data
    
    try:
        if Path('faq.index').exists():
            faq_data = pd.read_csv('faq.csv')
            logger.info("FAQ index loaded successfully")
            return True
        else:
            return initialize_faq_index()
    
    except Exception as e:
        logger.error(f"Error loading FAQ index: {e}")
        return False

@lru_cache(maxsize=100)
def get_relevant_faq_answers(review_text: str, k: int = 3) -> List[Dict[str, str]]:
    """Get relevant FAQ entries based on the review text"""
    global faq_data, question_list, encoder, faiss_index
    
    try:
        # Check if index needs initialization
        if not all([faq_data is not None, question_list, encoder, faiss_index]):
            if not initialize_faq_index():
                logger.warning("Failed to initialize FAQ index")
                return []
        
        # Get number of FAQs from FAISS index
        num_faqs = faiss_index.ntotal
        if num_faqs == 0:
            logger.warning("No FAQ entries in FAISS index")
            return []
        
        # Encode query
        query_embedding = encoder.encode([review_text], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().astype('float32')
        
        # Search
        k = min(k, num_faqs)
        distances, indices = faiss_index.search(query_embedding, k)
        
        # Process results
        relevant_faqs = []
        max_distance = 1.5  # Threshold for relevance
        
        for dist, idx in zip(distances[0], indices[0]):
            if dist > max_distance:
                continue
                
            if 0 <= idx < len(faq_data):
                relevant_faqs.append({
                    'question': faq_data.iloc[idx]['User Query'],
                    'answer': faq_data.iloc[idx]['Product Responses'],
                    'relevance': float(1.0 - (dist / 2.0))  # Convert distance to relevance
                })
        
        logger.info(f"Found {len(relevant_faqs)} relevant FAQs")
        return relevant_faqs
    
    except Exception as e:
        logger.error(f"Error in FAQ search: {str(e)}")
        return []

def get_llm_client():
    config = load_config()
    
    if config['llm_provider'] == 'openai':
        api_key = config['openai_api_key'] or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        return OpenAI(api_key=api_key)
    else:
        if not LLAMA_AVAILABLE:
            raise RuntimeError("LLaMA support is not available. Please install llama-cpp-python package.")
        return Llama(model_path=config['llama_model_path'])


def format_prompt_with_context(review_data: Dict[str, str], faq_context: str, 
                             feedback_history: Optional[List] = None, 
                             authenticity_analysis: Optional[Dict] = None, 
                             bot_analysis: Optional[Dict] = None) -> str:
    """
    Format the prompt with all available context for the LLM.
    """
    try:
        # Safely get the review text
        review_text = review_data.get('text', '')
        if not review_text:
            logger.warning("No review text provided in review_data")
            review_text = "No review text provided"

        # Format feedback history if available
        feedback_context = ""
        if feedback_history:
            feedback_entries = []
            for f in feedback_history:
                if isinstance(f, dict):
                    entry = (
                        f"Similar Review: {f.get('review_text', 'N/A')}\n"
                        f"Response: {f.get('response', 'N/A')}\n"
                        f"Feedback: {f.get('feedback', 'N/A')}\n"
                        f"Empathy Score: {f.get('empathy_score', 'N/A')}\n"
                        f"Hallucination: {'Yes' if f.get('hallucination_detected', False) else 'No'}\n"
                        f"Agent Notes: {f.get('agent_input', 'None')}\n"
                    )
                    feedback_entries.append(entry)
            if feedback_entries:
                feedback_context = "\nPrevious Feedback History:\n" + "\n".join(feedback_entries)

        # Format analysis context
        analysis_context = ""
        if authenticity_analysis or bot_analysis:
            analysis_context = "\nReview Analysis:\n"
            if authenticity_analysis and isinstance(authenticity_analysis, dict):
                analysis_context += (
                    f"Authenticity Score: {authenticity_analysis.get('authenticity_score', 'N/A')}\n"
                    f"Authenticity Confidence: {authenticity_analysis.get('confidence', 'N/A')}\n"
                    f"Red Flags: {', '.join(authenticity_analysis.get('red_flags', ['None']))}\n"
                )

            if bot_analysis and isinstance(bot_analysis, dict):
                analysis_context += (
                    f"Bot Detection Score: {bot_analysis.get('bot_score', 'N/A')}\n"
                    f"Bot Pattern Type: {bot_analysis.get('pattern_type', 'N/A')}\n"
                    f"Bot Indicators: {', '.join(bot_analysis.get('detection_points', ['None']))}\n"
                )

        # Combine all context
        prompt = f"""Review Analysis Context:{analysis_context}

FAQ Context:
{faq_context}
{feedback_context}

Review to respond to:
{review_text}

Consider all provided context when crafting your response. Pay special attention to:
1. Previous feedback and how it was received
2. Review authenticity and bot detection results
3. Relevant FAQ information

Please provide your response in two parts:
1. CONTEXT: Summarize the relevant context you're using from FAQs, feedback history, and analysis
2. RESPONSE: Your actual response to the review, considering all context

Format your response as:
CONTEXT: <your context summary>
RESPONSE: <your response to the review>
"""
        return prompt

    except Exception as e:
        logger.error(f"Error formatting prompt: {str(e)}")
        return f"Review to respond to:\n{review_data.get('text', 'No review text provided')}"

def analyze_review_authenticity(review_text):
    """
    Analyze if a review appears to be fake/spam based on various indicators.
    Returns a dict with authenticity score and reasoning.
    """
    try:
        prompt = f"""Analyze this review for authenticity. Consider these factors:
1. Generic/vague language
2. Extreme sentiment without specific details
3. Repetitive patterns or keywords
4. Unnatural language patterns
5. Inconsistencies in detail level
6. Bot-like characteristics
7. Marketing-style language
8. Time and location consistency
9. Reviewer behavior patterns

Review: {review_text}

Provide analysis in this format:
AUTHENTICITY_SCORE: (0-1, where 1 is most authentic)
RED_FLAGS: (list key suspicious elements)
CONFIDENCE: (0-1, how confident in this assessment)
REASONING: (detailed explanation)
"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in detecting fake reviews and spam content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        result = response.choices[0].message.content
        
        # Parse the response
        analysis = {}
        current_section = None
        sections = ['AUTHENTICITY_SCORE:', 'RED_FLAGS:', 'CONFIDENCE:', 'REASONING:']
        
        for line in result.split('\n'):
            for section in sections:
                if line.startswith(section):
                    current_section = section.replace(':', '').lower()
                    content = line.replace(section, '').strip()
                    if current_section == 'authenticity_score':
                        analysis[current_section] = float(content)
                    elif current_section == 'confidence':
                        analysis[current_section] = float(content)
                    elif current_section == 'red_flags':
                        analysis[current_section] = [flag.strip() for flag in content.strip('[]').split(',')]
                    else:
                        analysis[current_section] = content
                    break
            else:
                if current_section and current_section == 'reasoning':
                    analysis[current_section] = analysis.get(current_section, '') + ' ' + line.strip()

        return analysis
    except Exception as e:
        logger.exception("Error in authenticity analysis: %s", e)
        return {
            'authenticity_score': 0.5,
            'red_flags': ['Analysis failed'],
            'confidence': 0.0,
            'reasoning': f'Error during analysis: {str(e)}'
        }

def analyze_bot_patterns(review_text, review_metadata=None):
    """
    Analyze if a review was likely generated by a bot.
    Takes review text and optional metadata (timestamp, user info, etc.)
    Returns detailed bot analysis.
    """
    try:
        prompt = f"""Analyze this review for bot-generated content. Consider these specific indicators:

1. Language Patterns:
- Repetitive phrases or structures
- Unnatural language combinations
- Machine-like grammar perfection
- Templated content patterns

2. Content Analysis:
- Generic/non-specific details
- Contextually irrelevant information
- Inconsistent narrative flow
- Keyword stuffing

3. Bot Behavior Indicators:
- Time pattern anomalies
- Mass-produced content signs
- Cross-platform pattern matching
- Automated response characteristics

4. Technical Markers:
- NLP artifacts
- Language model patterns
- Statistical anomalies
- Content spinning signs

Review: {review_text}

Provide analysis in this format:
BOT_SCORE: (0-1, where 1 means definitely bot)
CONFIDENCE: (0-1)
DETECTION_POINTS: (list specific bot indicators found)
PATTERN_TYPE: (identify type of bot if detected: SPAM_BOT, REVIEW_BOT, AI_GENERATOR, HUMAN_LIKE_BOT, or UNCERTAIN)
REASONING: (detailed explanation)
"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in detecting bot-generated content and automated review systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        result = response.choices[0].message.content
        
        # Parse the response
        analysis = {}
        current_section = None
        sections = ['BOT_SCORE:', 'CONFIDENCE:', 'DETECTION_POINTS:', 'PATTERN_TYPE:', 'REASONING:']
        
        for line in result.split('\n'):
            for section in sections:
                if line.startswith(section):
                    current_section = section.replace(':', '').lower()
                    content = line.replace(section, '').strip()
                    if current_section in ['bot_score', 'confidence']:
                        analysis[current_section] = float(content)
                    elif current_section == 'detection_points':
                        analysis[current_section] = [point.strip() for point in content.strip('[]').split(',')]
                    elif current_section == 'pattern_type':
                        analysis[current_section] = content.strip()
                    else:
                        analysis[current_section] = content
                    break
            else:
                if current_section and current_section == 'reasoning':
                    analysis[current_section] = analysis.get(current_section, '') + ' ' + line.strip()

        return analysis
    except Exception as e:
        logger.exception("Error in bot analysis: %s", e)
        return {
            'bot_score': 0.5,
            'confidence': 0.0,
            'detection_points': ['Analysis failed'],
            'pattern_type': 'UNCERTAIN',
            'reasoning': f'Error during analysis: {str(e)}'
        }

def calculate_bot_detection_metrics(text: str) -> Dict[str, float]:
    """Calculate metrics for bot detection"""
    try:
        # Basic metrics
        char_count = len(text)
        word_count = len(text.split())
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Pattern analysis
        repeated_phrases = len(set([phrase for phrase in text.split() if text.count(phrase) > 2]))
        
        # Normalize scores between 0 and 1
        metrics = {
            'repetition_score': min(1.0, repeated_phrases / word_count) if word_count > 0 else 0,
            'length_score': min(1.0, word_count / 100),  # Normalize based on typical review length
            'complexity_score': min(1.0, avg_word_length / 10),  # Normalize based on typical word length
        }
        
        # Calculate overall bot probability
        metrics['bot_probability'] = (
            metrics['repetition_score'] * 0.4 +
            metrics['length_score'] * 0.3 +
            metrics['complexity_score'] * 0.3
        )
        
        return metrics
    except Exception as e:
        print(f"Error calculating bot metrics: {str(e)}")
        return {
            'repetition_score': 0,
            'length_score': 0,
            'complexity_score': 0,
            'bot_probability': 0
        }

def get_feedback_history(review_text, response_text=''):
    """
    Get relevant feedback history for similar reviews/responses.
    Returns list of relevant feedback entries.
    """
    try:
        feedback_file = Path('feedback_history.json')
        if not feedback_file.exists():
            return []

        with open(feedback_file, 'r') as f:
            feedback_history = json.load(f)
        
        # Get relevant feedback entries
        relevant_feedback = []
        for entry in feedback_history:
            # Calculate similarity scores
            review_similarity = fuzz.ratio(entry.get('review_text', ''), review_text)
            response_similarity = fuzz.ratio(entry.get('response', ''), response_text) if response_text else 0
            
            # If either review or response is similar enough, include the feedback
            if review_similarity > 60 or response_similarity > 60:
                entry['similarity_score'] = max(review_similarity, response_similarity)
                relevant_feedback.append(entry)
        
        # Sort by similarity score and return top 5
        relevant_feedback.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return relevant_feedback[:5]

    except Exception as e:
        logger.exception("Error getting feedback history: %s", e)
        return []

# Cache FAQ results to avoid repeated searches
@lru_cache(maxsize=1000)
def get_cached_faq_answers(text: str) -> Dict[str, Any]:
    return get_relevant_faq_answers(text)

# Cache sentiment analysis results
@lru_cache(maxsize=1000)
def get_cached_sentiment(text: str) -> Dict[str, Any]:
    return analyze_sentiment(text)

async def analyze_sentiment_async(text: str) -> Dict[str, Union[float, str, List[str]]]:
    """Analyze sentiment asynchronously"""
    try:
        sentiment_score = analyze_sentiment(text)
        return {
            'label': get_sentiment_label(sentiment_score),
            'score': sentiment_score,
            'aspects': extract_aspects(text)
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {'label': 'neutral', 'score': 0.5, 'aspects': []}

async def get_faq_context_async(text: str) -> str:
    """Get FAQ context asynchronously"""
    try:
        faq_entries = get_relevant_faq_answers(text)
        if not faq_entries:
            return ""
        
        # Format entries with relevance scores
        formatted_entries = []
        for entry in faq_entries:
            relevance_percent = int(entry['relevance'] * 100)
            formatted_entries.append(
                f"Similar Question: {entry['question']}\n"
                f"Answer: {entry['answer']}\n"
                f"Match Confidence: {relevance_percent}%"
            )
        
        return "\n\n".join(formatted_entries)
    except Exception as e:
        logger.error(f"Error getting FAQ context: {str(e)}")
        return ""

async def generate_response_async(text: str, sentiment: Dict, context: str) -> str:
    """Generate response asynchronously"""
    return generate_response(text, sentiment['score'], context)

async def parallel_analysis(text: str) -> Dict[str, Union[str, Dict]]:
    """Run analysis tasks in parallel with timeout"""
    try:
        async with asyncio.timeout(5.0):
            sentiment_score = analyze_sentiment(text)
            context = await get_faq_context_async(text)
            
            sentiment = {
                'label': get_sentiment_label(sentiment_score),
                'score': sentiment_score,
                'aspects': extract_aspects(text)
            }
            
            response = generate_response(text, sentiment_score, context)
            
            return {
                'sentiment': sentiment,
                'response': response,
                'context': context
            }
    except asyncio.TimeoutError:
        logger.error("Analysis timed out")
        return {
            'sentiment': {'label': 'neutral', 'score': 0.5, 'aspects': []},
            'response': "I apologize for the delay in processing your review.",
            'context': ""
        }
    except Exception as e:
        logger.error(f"Error in parallel analysis: {str(e)}")
        raise

def analyze_tone(text: str) -> Dict[str, float]:
    """Analyze the emotional tone of the text"""
    tone_indicators = {
        'frustrated': {'words': {'frustrated', 'annoying', 'waste', 'difficult', 'confusing', 'stuck'},
                      'score': 0},
        'urgent': {'words': {'asap', 'urgent', 'immediately', 'emergency', 'critical', 'deadline'},
                  'score': 0},
        'disappointed': {'words': {'disappointed', 'expected', 'should', 'but', 'however', 'unfortunately'},
                       'score': 0},
        'appreciative': {'words': {'thanks', 'appreciate', 'grateful', 'helpful', 'good', 'great'},
                        'score': 0}
    }
    
    words = text.lower().split()
    word_count = len(words)
    
    # Calculate tone scores
    for tone in tone_indicators.values():
        tone['score'] = sum(1 for word in words if word in tone['words']) / word_count
    
    return {tone: info['score'] for tone, info in tone_indicators.items()}

def extract_key_points(text: str) -> List[str]:
    """Extract key points from the review"""
    key_points = []
    
    # Feature-related keywords
    feature_keywords = {
        'performance': ['slow', 'fast', 'speed', 'quick', 'lag', 'responsive'],
        'usability': ['easy', 'difficult', 'intuitive', 'confusing', 'simple', 'complex'],
        'reliability': ['crash', 'bug', 'error', 'stable', 'reliable', 'broken'],
        'functionality': ['feature', 'work', 'doesn\'t work', 'broken', 'missing'],
        'support': ['help', 'support', 'service', 'assistance', 'contact']
    }
    
    words = text.lower().split()
    sentences = text.split('.')
    
    # Extract points based on feature keywords
    for category, keywords in feature_keywords.items():
        for keyword in keywords:
            if keyword in text.lower():
                # Find the relevant sentence
                for sentence in sentences:
                    if keyword in sentence.lower():
                        key_points.append({
                            'category': category,
                            'point': sentence.strip()
                        })
                        break
    
    return key_points

def generate_empathetic_response(text: str, sentiment_score: float, tone: Dict[str, float], 
                               key_points: List[Dict], context: str) -> str:
    """Generate an empathetic and contextual response"""
    
    # Base templates with empathy
    frustrated_templates = [
        "I understand how frustrating this must be for you. {}",
        "I can see why this situation would be frustrating. {}",
        "Your frustration is completely valid. {}"
    ]
    
    disappointed_templates = [
        "I understand this wasn't what you expected. {}",
        "I'm sorry we didn't meet your expectations. {}",
        "I appreciate you bringing this to our attention. {}"
    ]
    
    appreciative_templates = [
        "We're so glad to hear about your positive experience! {}",
        "Thank you for your kind words. {}",
        "We really appreciate your positive feedback! {}"
    ]
    
    neutral_templates = [
        "Thank you for sharing your thoughts. {}",
        "We appreciate your detailed feedback. {}",
        "Thank you for taking the time to provide this feedback. {}"
    ]
    
    # Select base template based on tone and sentiment
    if tone['frustrated'] > 0.2:
        base_template = random.choice(frustrated_templates)
    elif tone['disappointed'] > 0.2:
        base_template = random.choice(disappointed_templates)
    elif tone['appreciative'] > 0.2:
        base_template = random.choice(appreciative_templates)
    else:
        base_template = random.choice(neutral_templates)
    
    # Build response body based on key points
    response_points = []
    
    for point in key_points:
        category = point['category']
        if category == 'performance':
            response_points.append("Regarding the performance issues you mentioned, we're actively working on optimizations.")
        elif category == 'usability':
            response_points.append("We're constantly working to improve the user experience based on feedback like yours.")
        elif category == 'reliability':
            response_points.append("We take stability issues seriously and our team is investigating the reported problems.")
        elif category == 'functionality':
            response_points.append("We're reviewing the functionality concerns you've raised to ensure everything works as expected.")
        elif category == 'support':
            response_points.append("Our support team is here to help you with any additional assistance you need.")

    # Combine response elements
    response_body = " ".join(response_points)
    
    # Add action items or next steps
    if sentiment_score < 0.3:  # Negative sentiment
        action_items = "\n\nHere are some immediate steps we're taking:\n"
        action_items += "1. Our team will investigate the specific issues you've mentioned\n"
        action_items += "2. We'll reach out with updates on the reported problems\n"
        action_items += "3. We're prioritizing fixes for the concerns you've raised"
    else:
        action_items = ""

    # Add FAQ context if available
    faq_section = ""
    if context:
        faq_section = "\n\nYou might find these related answers helpful:\n" + context

    # Combine all parts
    full_response = base_template.format(response_body)
    if action_items:
        full_response += action_items
    if faq_section:
        full_response += faq_section
    
    return full_response

def process_review_task(
    review_text: str,
    empathy_level: int = 3,
    creativity_level: int = 3,
    include_personal_experience: bool = False,
    include_examples: bool = False,
    include_metaphors: bool = False
) -> Dict[str, Union[str, Dict]]:
    """Process a review with enhanced context and response generation"""
    try:
        # Get relevant FAQs using context manager
        relevant_faqs = context_manager.get_relevant_faqs(review_text)
        faq_context = context_manager.format_faq_context(relevant_faqs)
        
        # Analyze sentiment and authenticity
        sentiment_score = analyze_sentiment(review_text)
        authenticity_analysis = analyze_review_authenticity(review_text)
        bot_analysis = analyze_bot_patterns(review_text)
        
        # Get feedback history
        feedback_history = get_feedback_history(review_text)
        
        # Format prompt with all context
        prompt = format_prompt_with_context(
            {'text': review_text},  # Pass as dictionary with 'text' key
            faq_context,
            feedback_history,
            authenticity_analysis,
            bot_analysis
        )
        
        try:
            # Generate response using LLM
            llm_client = get_llm_client()
            completion = llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful customer service agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Properly access the response content
            response = completion.choices[0].message.content if completion.choices else "No response generated."
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            response = "I apologize, but I encountered an error generating a response."
        
        # Prepare the result dictionary with proper type checking
        result = {
            'sentiment': {
                'label': get_sentiment_label(sentiment_score),
                'score': float(sentiment_score),  # Ensure it's a float
            },
            'authenticity': authenticity_analysis if isinstance(authenticity_analysis, dict) else {},
            'bot_analysis': bot_analysis if isinstance(bot_analysis, dict) else {},
            'response': str(response),  # Ensure it's a string
            'context': str(faq_context),  # Ensure it's a string
            'faqs_used': [
                str(faq.get('question', '')) for faq in relevant_faqs 
                if isinstance(faq, dict)
            ]
        }
        
        # Log successful processing
        logger.info(f"Successfully processed review with {len(relevant_faqs)} relevant FAQs")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        # Return a safe fallback response
        return {
            'sentiment': {'label': 'neutral', 'score': 0.5},
            'authenticity': {},
            'bot_analysis': {},
            'response': "I apologize, but I encountered an error processing your review.",
            'context': "",
            'faqs_used': []
        }

def analyze_sentiment(text: str) -> float:
    """Enhanced sentiment analysis with better negative detection"""
    # Expanded word lists with informal expressions and intensifiers
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'awesome', 
        'fantastic', 'wonderful', 'best', 'superb', 'outstanding', 'helpful',
        'impressed', 'positive', 'recommended', 'happy', 'pleased', 'satisfied'
    }
    
    negative_words = {
        # Basic negative words
        'bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'horrible',
        'disappointing', 'useless', 'waste', 'difficult', 'unhappy',
        # Informal negative expressions
        'sucks', 'suck', 'garbage', 'trash', 'crap', 'rubbish', 'junk',
        'worthless', 'pathetic', 'joke', 'mess', 'disaster', 'fail', 'failed',
        # Problem indicators
        'broken', 'bug', 'buggy', 'error', 'issue', 'problem', 'glitch',
        'unusable', 'unreliable', 'unstable', 'annoying', 'frustrating'
    }
    
    # Intensifiers that strengthen sentiment
    intensifiers = {
        'very', 'really', 'extremely', 'absolutely', 'totally', 'completely',
        'utterly', 'seriously', 'literally', 'deeply', 'highly', 'super'
    }
    
    # Negation words that flip sentiment
    negations = {
        'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing', 'nowhere',
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
        "doesn't", "don't", "didn't", 'cannot', "can't", "couldn't"
    }

    text_lower = text.lower()
    words = text_lower.split()
    
    # Initialize scores
    pos_score = 0
    neg_score = 0
    
    # Check for negative phrases first (they take precedence)
    negative_phrases = [
        'waste of', 'not worth', 'get worse', 'getting worse',
        'gone downhill', 'going downhill', 'lost cause'
    ]
    for phrase in negative_phrases:
        if phrase in text_lower:
            neg_score += 2  # Strong negative weight for phrases
    
    # Analyze word by word with context
    for i, word in enumerate(words):
        prev_word = words[i-1] if i > 0 else ''
        next_word = words[i+1] if i < len(words)-1 else ''
        
        # Check for intensified sentiment
        intensity_multiplier = 1.5 if prev_word in intensifiers else 1.0
        
        # Check for negation (looking at previous words)
        negation_window = words[max(0, i-3):i]
        is_negated = any(neg in negation_window for neg in negations)
        
        if word in positive_words:
            if is_negated:
                neg_score += 1 * intensity_multiplier
            else:
                pos_score += 1 * intensity_multiplier
        elif word in negative_words:
            if is_negated:
                pos_score += 0.5 * intensity_multiplier  # Negated negative is slightly positive
            else:
                neg_score += 1.5 * intensity_multiplier  # Negative words carry more weight
        
        # Check for informal negative expressions (they carry extra weight)
        if word in {'sucks', 'garbage', 'trash', 'crap', 'rubbish'}:
            neg_score += 2  # These are strongly negative
    
    # Additional checks for overall text characteristics
    if '!' in text:
        # Exclamation marks intensify the dominant sentiment
        if neg_score > pos_score:
            neg_score *= 1.2
        elif pos_score > neg_score:
            pos_score *= 1.2
    
    # Calculate final sentiment score (0 to 1, where 0 is most negative)
    total_score = pos_score + neg_score
    if total_score == 0:
        return 0.5  # Neutral if no sentiment detected
    
    sentiment_score = pos_score / total_score
    
    # Adjust score based on overall text characteristics
    if any(phrase in text_lower for phrase in negative_phrases):
        sentiment_score *= 0.5  # Reduce score for negative phrases
    
    # Ensure the score stays within bounds
    return max(0.0, min(1.0, sentiment_score))

def get_sentiment_label(score: float) -> str:
    """Get more precise sentiment labels"""
    if score <= 0.2:
        return 'very negative'
    elif score <= 0.4:
        return 'negative'
    elif score <= 0.6:
        return 'neutral'
    elif score <= 0.8:
        return 'positive'
    else:
        return 'very positive'

def extract_aspects(text: str) -> List[str]:
    """Extract key aspects from text"""
    common_aspects = ['quality', 'price', 'service', 'delivery', 'features']
    return [aspect for aspect in common_aspects if aspect in text.lower()]

def generate_response(text: str, sentiment_score: float, context: str) -> str:
    """Generate more detailed and contextual response"""
    sentiment_label = get_sentiment_label(sentiment_score)
    
    # Enhanced response templates
    templates = {
        'positive': [
            "Thank you for your positive feedback! We're delighted to hear that you're satisfied with our service.",
            "We really appreciate your kind words and positive review!",
            "Thank you for the great review! We're glad we could meet your expectations."
        ],
        'negative': [
            "We apologize for any inconvenience you've experienced. Your feedback helps us improve.",
            "We're sorry to hear about your experience. We take your feedback seriously and will work on improvements.",
            "Thank you for bringing this to our attention. We apologize for not meeting your expectations."
        ],
        'neutral': [
            "Thank you for your feedback. We appreciate your balanced perspective.",
            "Thank you for taking the time to share your thoughts with us.",
            "We value your feedback and will take your comments into consideration."
        ]
    }
    
    # Select a template based on sentiment and add variety
    base_response = random.choice(templates[sentiment_label])
    
    # Add context if available
    if context:
        base_response += "\n\nBased on your review, here's some relevant information that might help:\n" + context
    
    return base_response

def format_faq_context(faqs: List[Dict[str, str]]) -> str:
    """Format FAQ entries into readable context"""
    if not faqs:
        return ""
    
    formatted = []
    for i, faq in enumerate(faqs, 1):
        relevance_percent = int(faq.get('relevance', 0) * 100)
        formatted.append(
            f"Related Question {i}:\n"
            f"Q: {faq.get('question', '')}\n"
            f"A: {faq.get('answer', '')}\n"
            f"Relevance: {relevance_percent}%"
        )
    
    return "\n\n".join(formatted)

class ContextManager:
    def __init__(self):
        self.faq_data = None
        self.question_list = None
        self.encoder = None
        self.faiss_index = None
        self.initialize()

    def initialize(self):
        """Initialize FAQ index and other resources"""
        try:
            # Load FAQ data from CSV
            csv_path = os.path.join('data', 'faq.csv')
            if not os.path.exists(csv_path):
                logger.warning(f"FAQ file not found at {csv_path}")
                return False
                
            self.faq_data = pd.read_csv(csv_path)
            logger.info(f"Loaded FAQ data with shape: {self.faq_data.shape}")
            
            # Store questions separately
            self.question_list = self.faq_data['User Query'].tolist()
            if not self.question_list:
                logger.warning("No questions found in FAQ data")
                return False
                
            logger.info(f"Number of questions loaded: {len(self.question_list)}")
            
            # Initialize sentence transformer
            self.encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            
            # Create embeddings
            embeddings = self.encoder.encode(self.question_list, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy().astype('float32')
            
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings)
            
            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error initializing FAQ index: {str(e)}")
            return False

    def get_relevant_faqs(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Get relevant FAQ entries based on the query text"""
        try:
            if not all([self.faq_data is not None, self.question_list, self.encoder, self.faiss_index]):
                if not self.initialize():
                    logger.warning("Failed to initialize FAQ index")
                    return []
            
            # Get number of FAQs from FAISS index
            num_faqs = self.faiss_index.ntotal
            if num_faqs == 0:
                logger.warning("No FAQ entries in FAISS index")
                return []
            
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy().astype('float32')
            
            # Search
            k = min(k, num_faqs)
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Process results
            relevant_faqs = []
            max_distance = 1.5  # Threshold for relevance
            
            for dist, idx in zip(distances[0], indices[0]):
                if dist > max_distance:
                    continue
                    
                if 0 <= idx < len(self.faq_data):
                    relevant_faqs.append({
                        'question': self.faq_data.iloc[idx]['User Query'],
                        'answer': self.faq_data.iloc[idx]['Product Responses'],
                        'relevance': float(1.0 - (dist / 2.0))  # Convert distance to relevance
                    })
            
            logger.info(f"Found {len(relevant_faqs)} relevant FAQs")
            return relevant_faqs
        
        except Exception as e:
            logger.error(f"Error in FAQ search: {str(e)}")
            return []

    def format_faq_context(self, faqs: List[Dict[str, str]]) -> str:
        """Format FAQ entries into readable context"""
        if not faqs:
            return ""
        
        formatted = []
        for i, faq in enumerate(faqs, 1):
            relevance_percent = int(faq.get('relevance', 0) * 100)
            formatted.append(
                f"Related Question {i}:\n"
                f"Q: {faq.get('question', '')}\n"
                f"A: {faq.get('answer', '')}\n"
                f"Relevance: {relevance_percent}%"
            )
        
        return "\n\n".join(formatted)

# Create a global instance of ContextManager
context_manager = ContextManager()

@celery.task
def process_review_task(
    review_text: str,
    empathy_level: int = 3,
    creativity_level: int = 3,
    include_personal_experience: bool = False,
    include_examples: bool = False,
    include_metaphors: bool = False
) -> Dict[str, Union[str, Dict]]:
    """Process a review with enhanced context and response generation"""
    try:
        # Get relevant FAQs using context manager
        relevant_faqs = context_manager.get_relevant_faqs(review_text)
        faq_context = context_manager.format_faq_context(relevant_faqs)
        
        # Analyze sentiment and authenticity
        sentiment_score = analyze_sentiment(review_text)
        authenticity_analysis = analyze_review_authenticity(review_text)
        bot_analysis = analyze_bot_patterns(review_text)
        
        # Get feedback history
        feedback_history = get_feedback_history(review_text)
        
        # Format prompt with all context
        prompt = format_prompt_with_context(
            {'text': review_text},  # Pass as dictionary with 'text' key
            faq_context,
            feedback_history,
            authenticity_analysis,
            bot_analysis
        )
        
        try:
            # Generate response using LLM
            llm_client = get_llm_client()
            completion = llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful customer service agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Properly access the response content
            response = completion.choices[0].message.content if completion.choices else "No response generated."
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            response = "I apologize, but I encountered an error generating a response."
        
        # Prepare the result dictionary with proper type checking
        result = {
            'sentiment': {
                'label': get_sentiment_label(sentiment_score),
                'score': float(sentiment_score),  # Ensure it's a float
            },
            'authenticity': authenticity_analysis if isinstance(authenticity_analysis, dict) else {},
            'bot_analysis': bot_analysis if isinstance(bot_analysis, dict) else {},
            'response': str(response),  # Ensure it's a string
            'context': str(faq_context),  # Ensure it's a string
            'faqs_used': [
                str(faq.get('question', '')) for faq in relevant_faqs 
                if isinstance(faq, dict)
            ]
        }
        
        # Log successful processing
        logger.info(f"Successfully processed review with {len(relevant_faqs)} relevant FAQs")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        # Return a safe fallback response
        return {
            'sentiment': {'label': 'neutral', 'score': 0.5},
            'authenticity': {},
            'bot_analysis': {},
            'response': "I apologize, but I encountered an error processing your review.",
            'context': "",
            'faqs_used': []
        }

def get_sentiment_label(score: float) -> str:
    """Get sentiment label from score"""
    if score <= 0.2:
        return 'very negative'
    elif score <= 0.4:
        return 'negative'
    elif score <= 0.6:
        return 'neutral'
    elif score <= 0.8:
        return 'positive'
    else:
        return 'very positive'

def format_faq_context(faqs: List[Dict[str, str]]) -> str:
    """Format FAQ entries into readable context"""
    if not faqs:
        return ""
    
    formatted = []
    for i, faq in enumerate(faqs, 1):
        relevance_percent = int(faq.get('relevance', 0) * 100)
        formatted.append(
            f"Related Question {i}:\n"
            f"Q: {faq.get('question', '')}\n"
            f"A: {faq.get('answer', '')}\n"
            f"Relevance: {relevance_percent}%"
        )
    
    return "\n\n".join(formatted)

def generate_response(
    text: str,
    sentiment_score: float,
    analysis: Dict[str, Any],
    faq_context: str,
    settings: Dict[str, Any]
) -> str:
    """Generate response based on analysis"""
    try:
        # Base templates with varying empathy levels
        templates = {
            1: "Thank you for your feedback. {}",
            2: "We appreciate your feedback. {}",
            3: "Thank you for sharing your thoughts. {}",
            4: "We truly appreciate you taking the time to share this. {}",
            5: "We're grateful you've shared this with us, and we completely understand how you feel. {}"
        }
        
        empathy_level = settings.get('empathy_level', 3)
        base_template = templates.get(empathy_level, templates[3])
        
        components = []
        
        # Add emotion-based response
        if analysis['primary_emotion'] in ['angry', 'frustrated']:
            components.append("We understand your frustration and we're here to help.")
        elif analysis['primary_emotion'] == 'disappointed':
            components.append("We apologize for not meeting your expectations.")
        
        # Add key points response
        if analysis['key_points']:
            for point in analysis['key_points']:
                components.append(f"Regarding your point about {point}, we're taking note of this feedback.")
        
        # Add FAQ context if available
        if faq_context:
            components.append("\nHere's some relevant information that might help:\n" + faq_context)
        
        # Combine all components
        response_body = " ".join(components)
        
        return base_template.format(response_body)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "We apologize, but we encountered an error generating the response."
