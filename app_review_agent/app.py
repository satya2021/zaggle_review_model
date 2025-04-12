# app.py
import os
from flask import Flask, render_template, request, jsonify
from celery import Celery
from dotenv import load_dotenv
import logging
import json
from collections import defaultdict
from openai import OpenAI
import pandas as pd
from werkzeug.utils import secure_filename
from pathlib import Path
from tasks import (
    load_faq_index,
    process_review_task,
    calculate_bot_detection_metrics,
    get_relevant_faq_answers,
    initialize_faq_index
)
from functools import lru_cache

# Make LLaMA import optional
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.config_from_object('celery_config')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

feedback_data = defaultdict(list)

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
        # Create default config file if it doesn't exist
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

# Initialize configuration on startup
if not CONFIG_FILE.exists():
    save_config(DEFAULT_CONFIG)

config = load_config()

# LLM client initialization
def get_llm_client():
    if config['llm_provider'] == 'openai':
        return OpenAI(api_key=config['openai_api_key'])
    else:
        if not LLAMA_AVAILABLE:
            raise RuntimeError("LLaMA support is not available. Please install llama-cpp-python package.")
        return Llama(model_path=config['llama_model_path'])

# Update the global client
llm_client = get_llm_client()

# Initialize FAQ index at startup
try:
    initialize_faq_index()
except Exception as e:
    app.logger.error(f"Error initializing FAQ index: {str(e)}")

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/api/dashboard/stats")
def dashboard_stats():
    # Return dashboard statistics
    stats = {
        'total_reviews': 100,  # Replace with actual stats
        'sentiment_distribution': {
            'positive': 50,
            'negative': 30,
            'neutral': 20
        }
    }
    return jsonify(stats)

@app.route('/process_review', methods=['POST'])
def process_review_endpoint():
    try:
        data = request.get_json()
        review_text = data.get('review_text')
        settings = data.get('settings', {})
        
        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400

        app.logger.info(f"Processing review with settings: {settings}")
        
        # Process the review with settings
        result = process_review_task(
            review_text=review_text,
            empathy_level=settings.get('empathyLevel', 3),
            creativity_level=settings.get('creativityLevel', 3),
            include_personal_experience=settings.get('includePersonalExperience', False),
            include_examples=settings.get('includeExamples', False),
            include_metaphors=settings.get('includeMetaphors', False)
        )
        
        # Log the results
        app.logger.info(f"FAQs used: {result.get('faqs_used', [])}")
        app.logger.info(f"Context length: {len(result.get('context', ''))}")
        
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error processing review: {str(e)}")
        return jsonify({
            'error': str(e),
            'sentiment': {'label': 'neutral', 'score': 0.5, 'tone': {}},
            'response': "I apologize, but I encountered an error processing your review.",
            'context': "",
            'faqs_used': []
        }), 500

@app.route("/task_status/<task_id>")
def task_status(task_id):
    task = celery.AsyncResult(task_id, app=celery)
    try:
        if task.state == 'PENDING':
            response = {
                'status': 'PENDING'
            }
        elif task.state != 'FAILURE':
            response = {
                'status': task.state,
                'result': task.info,
            }
        else:
            response = {
                'status': 'FAILURE',
                'error': str(task.info),
            }
        return jsonify(response)
    except Exception as e:
        logger.exception("Error checking task status for task_id: %s", task_id, exc_info=True)
        return jsonify({'status': 'FAILURE', 'error': f"Error retrieving task status: {e}"}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()
        review_text = data.get("review_text")
        generated_response = data.get("generated_response")
        feedback = data.get("feedback")
        hallucination_detected = data.get("hallucination_detected", False)
        empathy_score = data.get("empathy_score", 0)
        agent_input = data.get("agent_input")

        if not all([review_text, generated_response, feedback]):
            return jsonify({"status": "error", "message": "Missing data in feedback"}), 400

        feedback_key = f"{review_text}||{generated_response}"
        feedback_data[feedback_key].append({
            "feedback": feedback,
            "hallucination_detected": hallucination_detected,
            "empathy_score": empathy_score,
            "agent_input": agent_input
        })

        logger.info(f"Feedback received: Review: {review_text}, Response: {generated_response}, Feedback: {feedback}, Hallucination: {hallucination_detected}, Empathy: {empathy_score}, Agent Input: {agent_input}")

        return jsonify({"status": "success", "message": "Feedback received"}), 200

    except Exception as e:
        logger.exception("Error processing feedback: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": f"Error processing feedback: {e}"}), 500


@app.route("/admin/config", methods=["POST"])
def update_config():
    try:
        new_config = request.get_json()
        if new_config.get('llm_provider') == 'llama' and not LLAMA_AVAILABLE:
            return jsonify({
                "status": "error", 
                "message": "LLaMA support is not available. Please install llama-cpp-python package first."
            }), 400
        
        config.update(new_config)
        save_config(config)
        global llm_client
        llm_client = get_llm_client()
        return jsonify({"status": "success", "message": "Configuration updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/upload/reviews", methods=["POST"])
def upload_reviews():
    try:
        if 'reviews_file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['reviews_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            df = pd.read_csv(file)
            df.to_csv('app_reviews.csv', index=False)
            return jsonify({"status": "success", "message": f"Successfully uploaded {len(df)} reviews"})
        else:
            return jsonify({"status": "error", "message": "Invalid file format. Please upload a CSV file"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/upload/faqs", methods=["POST"])
def upload_faqs():
    try:
        if 'faq_file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['faq_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            df = pd.read_csv(file)
            df.to_csv('faq.csv', index=False)
            # Reload FAQ index
            celery.send_task('tasks.load_faq_index')
            return jsonify({"status": "success", "message": f"Successfully uploaded {len(df)} FAQs"})
        else:
            return jsonify({"status": "error", "message": "Invalid file format. Please upload a CSV file"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        review_text = data.get('review_text')
        response = data.get('response')
        feedback = {
            'is_good': data.get('is_good', False),
            'empathy_score': data.get('empathy_score'),
            'relevance_score': data.get('relevance_score'),
            'helpfulness_score': data.get('helpfulness_score'),
            'settings': data.get('settings', {}),
            'comments': data.get('comments')
        }
        
        if feedback_manager.save_feedback(review_text, response, feedback):
            return jsonify({'status': 'success', 'message': 'Feedback recorded successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save feedback'}), 500
    except Exception as e:
        app.logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    try:
        # Initialize FAQ index
        if not load_faq_index():
            logger.critical("Failed to initialize FAQ index. Application may not function correctly.")
        app.run(debug=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
