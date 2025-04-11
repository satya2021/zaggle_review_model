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


load_dotenv()

celery = Celery(__name__, broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
                backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))
celery.config_from_object('celery_config')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faq.index')

openai_client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
sentiment_analyzer = SentimentIntensityAnalyzer()

logger = logging.getLogger(__name__)


# --- Helper Functions ---
def load_faq_index():
    global faq_index
    try:
        logger.info("Attempting to load FAISS index from: %s", FAISS_INDEX_PATH)
        faq_index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info("FAISS index loaded successfully.")
    except RuntimeError as e:
        logger.error("Error loading FAISS index: %s. Creating a new index.", e)
        faq_index = create_faq_index()
    except Exception as e:
        logger.exception("Unexpected error during FAISS index loading: %s", e, exc_info=True)
        raise


def create_faq_index():
    """Loads FAQ data and creates the FAISS index."""
    global faq_index, faq_df
    try:
        logger.info("Creating a new FAISS index.")
        FAQ_CSV_PATH = os.environ.get('FAQ_CSV_PATH', 'Chatbot FAQ\'s.csv')
        faq_df = pd.read_csv(FAQ_CSV_PATH)
        faq_df.columns = faq_df.columns.str.strip()
        faq_embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(faq_df['User Query'].fillna('').tolist())
        dimension = faq_embeddings[0].shape[0]
        faq_index = faiss.IndexFlatL2(dimension)
        faq_index.add(np.array(faq_embeddings).astype('float32'))
        faiss.write_index(faq_index, FAISS_INDEX_PATH)
        logger.info("FAISS index created and saved to: %s", FAISS_INDEX_PATH)
        return faq_index
    except Exception as e:
        logger.exception("Error creating FAISS index: %s", e)
        raise


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'delay': 2, 'backoff': True})
def process_review_task(self, review_data):
    try:
        review_text = review_data.get("text", "")
        rating = review_data.get("rating", None)
        feedback_requested = review_data.get("feedback_requested", False)

        sentiment = analyze_sentiment(review_text)
        sentiment = verify_sentiment_with_llm(review_text, sentiment)

        persona = select_persona(sentiment, rating)
        relevant_answers = get_relevant_faq_answers(review_text)
        generated_response = generate_response(review_text, relevant_answers, sentiment, persona=persona)

        return {"generated_response": generated_response, "sentiment": sentiment, "feedback_requested": feedback_requested}

    except Exception as exc:
        logger.error(f"Task {self.name} failed: {exc}")
        raise


def get_relevant_faq_answers(review_text, top_k=3):
    try:
        review_embedding = embedding_model.encode([review_text])
        if np.any(np.isnan(review_embedding)) or np.any(np.isinf(review_embedding)):
            logger.warning("Embedding contains NaN or Inf! Returning empty list.")
            return []
        D, I = faq_index.search(review_embedding.astype('float32'), top_k)
        relevant_answers = [faq_df.iloc[i]['Product Responses'] for i in I[0]]
        return relevant_answers
    except Exception as e:
        logger.exception("Error in get_relevant_faq_answers: %s", e)
        return []


def analyze_sentiment(review_text):
    try:
        scores = sentiment_analyzer.polarity_scores(review_text)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        logger.exception("Error in analyze_sentiment: %s", e)
        return "neutral"


def clean_text(text):
    try:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    except Exception as e:
        logger.exception("Error in clean_text: %s", e)
        return text


def generate_response(review_text, relevant_answers, sentiment, persona="helpful_assistant",
                      hallucination_control="strict", empathy_level="medium", agent_input=""):
    """
    Generates a response to a user review.

    Args:
        review_text: The text of the user review.
        relevant_answers: Relevant FAQ answers retrieved from the knowledge base.
        sentiment: The sentiment of the review (positive, negative, or neutral).
        persona: The desired persona of the response (e.g., "helpful_assistant", "empathetic_support").
        hallucination_control: Level of strictness in avoiding hallucination ("strict", "moderate", "lenient").
        empathy_level: Level of empathy to express ("high", "medium", "low").
        agent_input: Any input from the agent (e.g., edited response).

    Returns:
        The generated response text.
    """
    try:
        cleaned_review_text = clean_text(review_text)

        if persona == "helpful_assistant":
            persona_instructions = """You are a helpful and informative customer support assistant for a mobile app.
            Provide clear and concise answers, and guide the user towards solutions."""
        elif persona == "empathetic_support":
            empathy_phrases = {
                "high": ["I understand how frustrating this must be.", "I sincerely apologize for the inconvenience.", "We appreciate your patience.", "I'm here to help."],
                "medium": ["I understand this can be annoying.", "Sorry for the trouble.", "Thanks for your patience.", "Let's see what we can do."],
                "low": ["Okay, I see.", "Here's what you can do.", "Thank you.", "I can assist with that."]
            }
            chosen_phrases = empathy_phrases.get(empathy_level, empathy_phrases["medium"])
            persona_instructions = f"""You are an empathetic and understanding customer support agent.
            Acknowledge the user's frustration, offer sincere apologies, and provide solutions.
            Use phrases like: {", ".join(chosen_phrases)}"""
        elif persona == "feature_promoter":
            persona_instructions = """You are a marketing-oriented assistant.
            Enthusiastically highlight the app's features and encourage user engagement. Keep the tone positive and upbeat."""
        else:
            persona_instructions = """You are a customer support agent.
            Provide helpful and concise information."""

        hallucination_prompt = ""
        if hallucination_control == "strict":
            hallucination_prompt = "You MUST only use information provided in the FAQ answers. Do not invent or assume any information."
        elif hallucination_control == "moderate":
            hallucination_prompt = "Prefer information from the FAQ answers, but you may add general knowledge if absolutely necessary."
        elif hallucination_control == "lenient":
            hallucination_prompt = "Use the FAQ answers as a guide, but you have flexibility to provide additional information."

        prompt = f"""
        {persona_instructions}
        {hallucination_prompt}

        Review: "{cleaned_review_text}"
        Sentiment: {sentiment}

        Relevant FAQ Answers:
        { "\\n".join(relevant_answers) }

        Generate a concise and on-brand response (maximum 200 tokens).
        If the sentiment is negative, address the user's concerns using information from the FAQ answers and, if possible, provide solutions.
        If the sentiment is positive (4/5 stars), thank the user and suggest exploring app features.
        If the FAQ does not contain the answer, respond with "Please contact our support team at care@zaggle.in for further assistance."
        """

        if agent_input:
            prompt = f"Rewrite the following response to be better: {agent_input}. Original response: {prompt}"

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Error generating response: %s", e)
        return "I'm sorry, I encountered an error. Please contact our support team at care@zaggle.in for further assistance."


def verify_sentiment_with_llm(review_text, vader_sentiment):
    try:
        llm_prompt = f"Review: '{review_text}'. VADER says: {vader_sentiment}. Is the overall sentiment positive, negative, or neutral? Answer only with one of these three words."
        llm_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=20
        )
        llm_response_text = llm_response.choices[0].message.content.strip().lower()
        if llm_response_text in ["positive", "negative", "neutral"]:
            return llm_response_text
        else:
            return vader_sentiment  # Fallback to VADER
    except Exception as e:
        logger.exception("LLM Sentiment Verification Error: %s", e)
        return vader_sentiment


def select_persona(sentiment, rating):
    """
    Selects the appropriate persona based on sentiment and rating.
    """
    if sentiment == "negative":
        return "empathetic_support"
    elif sentiment == "positive" and rating in [4, 5]:
        return "feature_promoter"
    else:
        return "helpful_assistant"
