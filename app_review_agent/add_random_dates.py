import pandas as pd
import random
from datetime import datetime, timedelta
import os

def add_random_dates():
    # Define column names
    columns = [
        'package', 'version', 'build', 'language', 'device',
        'review_date', 'review_timestamp', 'update_date', 'update_timestamp',
        'rating', 'title', 'text', 'response_date', 'response_timestamp',
        'response', 'url', 'review_id'
    ]
    
    # Read the existing CSV file with proper settings
    df = pd.read_csv(
        'app_reviews.csv',
        names=columns,
        quoting=1,  # QUOTE_ALL
        escapechar='\\',
        na_values=['', 'nan', 'NaN'],
        keep_default_na=True,
        on_bad_lines='skip'  # Skip problematic lines
    )
    
    # Get the current date
    current_date = datetime.now()
    
    # Generate random dates for each review
    for i in range(len(df)):
        try:
            # Generate a random number of days between 0 and 365 (1 year)
            random_days = random.randint(0, 365)
            
            # Calculate the random date
            random_date = current_date - timedelta(days=random_days)
            
            # Format the date as YYYY-MM-DD
            formatted_date = random_date.strftime('%Y-%m-%d')
            
            # Update the review_date and update_date
            df.at[i, 'review_date'] = formatted_date + 'T' + random_date.strftime('%H:%M:%S') + 'Z'
            df.at[i, 'update_date'] = formatted_date + 'T' + random_date.strftime('%H:%M:%S') + 'Z'
            
            # Update timestamps
            timestamp_ms = str(int(random_date.timestamp() * 1000))
            df.at[i, 'review_timestamp'] = timestamp_ms
            df.at[i, 'update_timestamp'] = timestamp_ms
            
            # If there's a response, set response_date to review_date + random days (1-7)
            if pd.notna(df.at[i, 'response']) and str(df.at[i, 'response']).strip():
                response_days = random.randint(1, 7)
                response_date = random_date + timedelta(days=response_days)
                df.at[i, 'response_date'] = response_date.strftime('%Y-%m-%d') + 'T' + response_date.strftime('%H:%M:%S') + 'Z'
                df.at[i, 'response_timestamp'] = str(int(response_date.timestamp() * 1000))
            else:
                df.at[i, 'response_date'] = ''
                df.at[i, 'response_timestamp'] = ''
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue
    
    try:
        # Save the updated DataFrame back to CSV with proper quoting
        df.to_csv('app_reviews.csv', index=False, quoting=1, escapechar='\\')
        print("Successfully added random dates to app_reviews.csv")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    add_random_dates() 