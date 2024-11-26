from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_hashtags():
    user_tweet = request.form.get('tweet', '')
    hashtags = ['#Example', '#Hashtag', '#Generation']  # Placeholder for hashtag generation logic
    return render_template('index.html', tweet=user_tweet, hashtags=hashtags)

if __name__ == '__main__':
    app.run(debug=True)
