# App Review Analysis Dashboard

A Flask-based web application for analyzing and visualizing app reviews, with sentiment analysis and response generation capabilities.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd app_review_agent
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the root directory with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
CELERY_BROKER_URL=redis://localhost:6379/0
```

2. Create the following directories if they don't exist:
```bash
mkdir -p data
mkdir -p config
```

## Data Requirements

### App Reviews CSV (`app_reviews.csv`)

The application expects a CSV file with the following columns:

```csv
package,version,build,language,device,review_date,review_timestamp,update_date,update_timestamp,rating,title,text,response_date,response_timestamp,response,url,review_id
```

#### Column Descriptions:
- `package`: App package name (e.g., com.example.app)
- `version`: App version number (e.g., 1.0.0)
- `build`: Build number (e.g., 123)
- `language`: Review language code (e.g., en, es, fr)
- `device`: Device model (e.g., iPhone 12, Samsung Galaxy S21)
- `review_date`: Date of review in YYYY-MM-DD format
- `review_timestamp`: Unix timestamp in milliseconds
- `update_date`: Date of review update in YYYY-MM-DD format
- `update_timestamp`: Unix timestamp in milliseconds
- `rating`: Star rating (1-5)
- `title`: Review title
- `text`: Review content
- `response_date`: Date of response in YYYY-MM-DD format
- `response_timestamp`: Unix timestamp in milliseconds
- `response`: Developer response text
- `url`: Review URL
- `review_id`: Unique identifier for the review

#### Example Rows:
```csv
com.example.app,1.0.0,123,en,iPhone 12,2024-03-15,1710504000000,2024-03-15,1710504000000,5,Great app!,This app is amazing!,2024-03-15,1710504005000,Thank you for your feedback!,https://play.google.com/store/apps/details?id=com.example.app,123e4567-e89b-12d3-a456-426614174000
com.example.app,1.0.0,123,es,Samsung Galaxy S21,2024-03-14,1710417600000,2024-03-14,1710417600000,3,Regular,La aplicación funciona bien pero podría mejorar.,2024-03-14,1710417605000,Gracias por tu comentario. Trabajaremos en mejorar la experiencia.,https://play.google.com/store/apps/details?id=com.example.app,123e4567-e89b-12d3-a456-426614174001
com.example.app,1.0.0,123,en,iPhone 13,2024-03-13,1710331200000,2024-03-13,1710331200000,1,Not working,App crashes on startup. Very disappointed.,2024-03-13,1710331205000,We apologize for the inconvenience. Please try clearing the app cache and reinstalling.,https://play.google.com/store/apps/details?id=com.example.app,123e4567-e89b-12d3-a456-426614174002
```

### Feedback CSV (`data/feedback.csv`)

The application expects a CSV file with the following columns:

```csv
review_id,timestamp,review_text,response,empathy_score,relevance_score,helpfulness_score,is_good,comments
```

#### Column Descriptions:
- `review_id`: Matches the review_id from app_reviews.csv
- `timestamp`: ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS.sssZ)
- `review_text`: Original review text
- `response`: Developer response text
- `empathy_score`: Score from 1-5
- `relevance_score`: Score from 1-5
- `helpfulness_score`: Score from 1-5
- `is_good`: Boolean (true/false)
- `comments`: Additional feedback comments

#### Example Rows:
```csv
123e4567-e89b-12d3-a456-426614174000,2024-03-15T12:00:00.000Z,This app is amazing!,Thank you for your feedback!,4,5,5,true,Great response!
123e4567-e89b-12d3-a456-426614174001,2024-03-14T15:30:00.000Z,La aplicación funciona bien pero podría mejorar.,Gracias por tu comentario. Trabajaremos en mejorar la experiencia.,3,4,3,true,Response could be more specific about improvements.
123e4567-e89b-12d3-a456-426614174002,2024-03-13T09:45:00.000Z,App crashes on startup. Very disappointed.,We apologize for the inconvenience. Please try clearing the app cache and reinstalling.,2,3,2,false,Response doesn't address the core issue.
```

### Data Validation Rules

1. App Reviews CSV:
   - All timestamps must be in milliseconds
   - Dates must be in YYYY-MM-DD format
   - Ratings must be integers between 1 and 5
   - Review IDs must be unique
   - URLs must be valid and complete

2. Feedback CSV:
   - Timestamps must be in ISO 8601 format
   - Scores must be integers between 1 and 5
   - is_good must be either 'true' or 'false'
   - review_id must match an existing review in app_reviews.csv

## Running the Application

1. Start Redis server (required for Celery):
```bash
redis-server
```

2. Start the Celery worker in a new terminal:
```bash
celery -A app.celery worker --loglevel=info
```

3. Start the Flask application:
```bash
python3 app.py
```

The application will be available at `http://localhost:5000`

## Features

- Dashboard with key metrics
- Sentiment analysis of reviews
- Response generation
- Feedback tracking
- Detailed analysis view
- Admin configuration

## API Endpoints

- `/api/dashboard/stats` - Get dashboard statistics
- `/api/analysis/stats` - Get detailed analysis statistics
- `/process_review` - Process a new review
- `/feedback` - Submit feedback on responses
- `/admin` - Admin configuration interface

## Data Processing

The application processes reviews in the following way:

1. Reviews are loaded from `app_reviews.csv`
2. Each review undergoes sentiment analysis
3. Feedback data is merged with reviews
4. Metrics are calculated for display
5. Responses are generated using AI models

## Error Handling

The application includes robust error handling for:
- Missing or malformed CSV files
- Invalid data types
- API failures
- Database connection issues

## Troubleshooting

Common issues and solutions:

1. "Command not found: python"
   - Use `python3` instead of `python`
   - Ensure Python is installed and in PATH

2. CSV parsing errors
   - Ensure CSV files follow the required format
   - Check for missing or extra columns
   - Verify data types in each column

3. API errors
   - Check OpenAI API key in `.env`
   - Verify internet connection
   - Check API rate limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license] 