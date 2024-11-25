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

def load_data(file_path):
  """Load the dataset."""
  try:
      # Open the file with the appropriate encoding and handle errors in the open() method
      with open(file_path, encoding="utf-8", errors="replace") as f:
          df = pd.read_csv(f)  # Now reading the file into a DataFrame
  except FileNotFoundError:
      print(f"Error: The file at {file_path} was not found.")
      exit()
  except Exception as e:
      print(f"Error reading the file: {e}")
      exit()
  return df

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
    """Clean text by removing URLs, mentions, hashtags, punctuation, stop words, specific words ('username', 'url'), and applying lemmatization."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
    text = remove_emojis(text)  # Remove emojis
    text = " ".join([word for word in text.split() if word not in stop_words and word not in {"username", "url","via"}])  # Remove stop words and specific words
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
    return text



def preprocess_file(file_path, text_column, output_dir):
    """Preprocess a single CSV file and add a 'topic' column."""
    print(f"Processing file: {file_path}")

    # Extract the topic from the filename (remove '.csv')
    topic_name = os.path.basename(file_path).replace('.csv', '')

    # Load the file
    df = pd.read_csv(file_path)

    # Filter for English text only
    print("Filtering non-English tweets...")
    df = df[df[text_column].apply(filter_non_english)]

    # Clean the text column
    print("Cleaning text data...")
    df['cleaned_text'] = df[text_column].apply(clean_text)

    # Drop the original text column
    df.drop(text_column, axis=1, inplace=True)
    df.drop("tags", axis=1, inplace=True)

    # Add the 'topic' column
    df['topic'] = topic_name

    # Save the preprocessed file
    output_file = os.path.join(output_dir,
                               f"preprocessed_{os.path.basename(file_path)}")
    df.to_csv(output_file, index=False)
    print(f"File saved: {output_file}")

    return df


def preprocess_folder(input_folder, text_column, output_folder,
    combined_file_name="combined_preprocessed.csv"):
    """Preprocess all CSV files in a folder and save a combined CSV."""
    print(f"Processing folder: {input_folder}")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List to store individual DataFrames
    combined_dfs = []

    # Process each CSV file in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            df = preprocess_file(file_path, text_column, output_folder)
            combined_dfs.append(df)  # Add the resulting DataFrame to the list

    # Combine all DataFrames into one
    print("Combining all preprocessed DataFrames...")
    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Shuffle the combined DataFrame
    print("Shuffling the combined DataFrame...")
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(
        drop=True)

    # Save the combined DataFrame to a CSV file
    combined_file_path = os.path.join(output_folder, combined_file_name)
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Combined and shuffled file saved: {combined_file_path}")
