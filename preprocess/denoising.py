import pandas as pd
import spacy
import re
import enchant
import sys
import numpy as np
sys.path.append('../preprocess')
from preprocess import load_data


# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")
d = enchant.Dict("en_US")

# Define the slang removal function
def remove_slang(text):
    """Remove slang and words with repeated characters or excessive length."""
    # Remove words that are longer than a certain length or contain repeated characters
    cleaned_text = re.sub(r'\b(\w)\1{2,}\b', '', text)  # Remove repeated characters
    cleaned_text = ' '.join([word for word in cleaned_text.split() if len(word) <= 15])  # Remove overly long words
    return cleaned_text

def remove_non_english(text):
    """Remove non-English words and gibberish."""
    words = text.split()
    cleaned_words = [word for word in words if d.check(re.sub(r'[^a-zA-Z]', '', word)) and len(word) > 2]
    return ' '.join(cleaned_words)


# Define the preprocessing pipeline function
def preprocess_pipeline(text):
    """Clean the text using a predefined set of rules."""
    # Step 1: Remove slang
    cleaned_text = remove_slang(text)
    cleaned_text = remove_non_english(cleaned_text)
    # Step 2: Process the text using SpaCy
    doc = nlp(cleaned_text)

    # Step 3: Extract nouns and proper nouns
    nouns = [token.text for token in doc if
             token.pos_ in ['NOUN', 'PROPN']]  # NOUN for common nouns, PROPN for proper nouns

    # Return the extracted nouns as a string
    return ' '.join(nouns)

file_path = "../data/preprocessed/combined_preprocessed.csv"

# Read the CSV file into a pandas DataFrame
df = load_data(file_path)

# Apply the preprocessing pipeline to the text column (adjust the column name as necessary)
df['processed_text'] = df['cleaned_text'].apply(preprocess_pipeline)  # Assuming 'text' is the column name

# Save the new DataFrame to a new CSV file
df.to_csv("../data/preprocessed/combined_preprocessed_denoised.csv", index=False)

print("Preprocessing complete. The cleaned data is saved as 'combined_preprocessed_denoised.csv'.")