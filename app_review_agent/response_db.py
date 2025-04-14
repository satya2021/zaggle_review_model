import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class ResponseDatabase:
    def __init__(self, db_file: str = 'data/response_database.csv'):
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(exist_ok=True)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectors = None
        self.responses = []
        self.load_database()

    def load_database(self):
        """Load responses from CSV file"""
        if self.db_file.exists():
            try:
                df = pd.read_csv(self.db_file)
                self.responses = df.to_dict('records')
                if self.responses:
                    # Create vectors for similarity matching
                    texts = [r['review_text'] for r in self.responses]
                    self.vectors = self.vectorizer.fit_transform(texts)
            except Exception as e:
                logger.error(f"Error loading response database: {e}")
                self.responses = []
                self.vectors = None

    def save_response(self, review_text: str, response: str, feedback_data: Dict[str, Any]):
        """Save a new response with its feedback"""
        try:
            new_entry = {
                'review_text': review_text,
                'response': response,
                'empathy_score': feedback_data.get('empathy_score', 0),
                'relevance_score': feedback_data.get('relevance_score', 0),
                'helpfulness_score': feedback_data.get('helpfulness_score', 0),
                'is_good': feedback_data.get('is_good', False),
                'use_count': 0,
                'timestamp': datetime.now().isoformat()
            }

            # Only save responses with good feedback
            if new_entry['is_good'] and all(new_entry[k] >= 3 for k in ['empathy_score', 'relevance_score', 'helpfulness_score']):
                self.responses.append(new_entry)
                
                # Update vectors
                texts = [r['review_text'] for r in self.responses]
                self.vectors = self.vectorizer.fit_transform(texts)
                
                # Save to CSV
                df = pd.DataFrame(self.responses)
                df.to_csv(self.db_file, index=False)
                logger.info(f"Saved new response to database. Total responses: {len(self.responses)}")

        except Exception as e:
            logger.error(f"Error saving response: {e}")

    def find_similar_responses(self, review_text: str, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """Find similar responses from the database"""
        if not self.responses or self.vectors is None:
            return []

        try:
            # Transform new review text
            review_vector = self.vectorizer.transform([review_text])
            
            # Calculate similarities
            similarities = cosine_similarity(review_vector, self.vectors)[0]
            
            # Get responses above similarity threshold
            similar_responses = []
            for idx, similarity in enumerate(similarities):
                if similarity >= min_similarity:
                    response = self.responses[idx].copy()
                    response['similarity'] = float(similarity)
                    similar_responses.append(response)
            
            # Sort by similarity and feedback scores
            similar_responses.sort(key=lambda x: (
                x['similarity'],
                x['helpfulness_score'],
                x['relevance_score'],
                x['empathy_score']
            ), reverse=True)
            
            return similar_responses[:3]  # Return top 3 similar responses

        except Exception as e:
            logger.error(f"Error finding similar responses: {e}")
            return []

    def get_response_stats(self) -> Dict[str, Any]:
        """Get statistics about stored responses"""
        if not self.responses:
            return {
                'total_responses': 0,
                'avg_helpfulness': 0,
                'avg_empathy': 0,
                'avg_relevance': 0,
                'reuse_rate': 0
            }

        total = len(self.responses)
        return {
            'total_responses': total,
            'avg_helpfulness': sum(r['helpfulness_score'] for r in self.responses) / total,
            'avg_empathy': sum(r['empathy_score'] for r in self.responses) / total,
            'avg_relevance': sum(r['relevance_score'] for r in self.responses) / total,
            'reuse_rate': sum(r['use_count'] for r in self.responses) / total if total > 0 else 0
        } 