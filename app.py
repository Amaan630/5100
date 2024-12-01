import os
import pandas as pd
from flask import Flask, render_template, request

from vectorization.vectorizer import HashtagRecommender, Vectorizer

app = Flask(__name__)

DATASET_FOLDER = "data/data/preprocessed/"

@app.route('/')
def home():
    return render_template('home.html', tweet='', topics=[], hashtags=[])

@app.route('/generate', methods=['POST'])
def generate_hashtags():
    user_tweet = request.form.get('tweet', '')

    topics = ['Arts', 'Culture', 'Poetry'] if user_tweet else []
    recommender = HashtagRecommender()
    recommender.initialize()
    hashtags = recommender.get_top_hashtags(user_tweet)

    return render_template('home.html', tweet=user_tweet, topics=topics, hashtags=hashtags)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/datasets')
def datasets():
    datasets = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.csv')]
    return render_template('datasets.html', datasets=datasets)

@app.route('/datasets/<filename>')
def dataset_detail(filename):
    file_path = os.path.join(DATASET_FOLDER, filename)
    try:
        # Load CSV and display first few rows
        df = pd.read_csv(file_path)
        data_preview = df.head(10).to_dict(orient='records')  # Convert first 10 rows to a list of dicts
        columns = df.columns.tolist()
        return render_template('dataset_detail.html', filename=filename, columns=columns, rows=data_preview)
    except Exception as e:
        error_message = f"Error loading dataset {filename}: {str(e)}"
        return render_template('dataset_detail.html', filename=filename, columns=[], rows=[], error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
    