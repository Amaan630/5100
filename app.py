from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', tweet='', hashtags=[])

@app.route('/generate', methods=['POST'])
def generate_hashtags():
    user_tweet = request.form.get('tweet', '')
    
    # Placeholder logic for generating hashtags
    if user_tweet:
        words = user_tweet.split()
        hashtags = [f"#{word.capitalize()}" for word in words[:3]]  # Generate hashtags from first 3 words
    else:
        hashtags = ['#Example', '#Hashtag', '#Generation']

    return render_template('home.html', tweet=user_tweet, hashtags=hashtags)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/datasets')
def datasets():
    datasets = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
    return render_template('datasets.html', datasets=datasets)

if __name__ == '__main__':
    app.run(debug=True)
    