# app.py
import os
from flask import Flask, render_template, request, jsonify
from celery import Celery
from dotenv import load_dotenv
import logging
import json
from collections import defaultdict
from openai import OpenAI

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


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_review", methods=["POST"])
def process_review():
    data = request.get_json()
    review_data = data.get("review_data", {})
    task = celery.send_task('tasks.process_review_task', args=[review_data])
    return jsonify({"task_id": task.id, "status": "processing", "feedback_requested": review_data.get("request_feedback", False)})


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


if __name__ == "__main__":
    from tasks import load_faq_index

    try:
        load_faq_index()
    except Exception as e:
        logger.critical("Failed to load FAISS index. Application may not function correctly. Error: %s", e, exc_info=True)
    app.run(debug=True)
