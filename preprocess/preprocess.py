import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from collections import Counter
import emoji

# Download necessary nltk data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Define preprocessing functions
def remove_emojis(text):
    """Remove emojis from text."""
    return emoji.replace_emoji(text, "")

def filter_non_english(text):
    """Filter out non-English tweets."""
    try:
        return detect(text) == "en"
    except:
        return False

def clean_text(text):
    """Clean text by removing URLs, mentions, hashtags, punctuation, stop words, and applying lemmatization."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
    text = remove_emojis(text)  # Remove emojis
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stop words
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
    return text

def preprocess_file(file_path, text_column, output_dir):
    """Preprocess a single CSV file."""
    print(f"Processing file: {file_path}")
    # Load the file
    df = pd.read_csv(file_path)

    # Filter for English text only
    print("Filtering non-English tweets...")
    df = df[df[text_column].apply(filter_non_english)]

    # Clean the text column
    print("Cleaning text data...")
    df['cleaned_text'] = df[text_column].apply(clean_text)

    # Save the preprocessed file
    output_file = os.path.join(output_dir, f"preprocessed_{os.path.basename(file_path)}")
    df.to_csv(output_file, index=False)
    print(f"File saved: {output_file}")

def preprocess_folder(input_folder, text_column, output_folder):
    """Preprocess all CSV files in a folder."""
    print(f"Processing folder: {input_folder}")
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each CSV file in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            preprocess_file(file_path, text_column, output_folder)