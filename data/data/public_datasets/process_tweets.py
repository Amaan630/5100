import os
import re

import pandas as pd


def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    if not hashtags:
        return None
    # Remove the # symbol and join with spaces
    return ' '.join(tag[1:] for tag in hashtags)

def clean_text(text, hashtags):
    # Remove all hashtags from the text
    for tag in hashtags.split():
        text = text.replace(f'#{tag}', '').strip()
    return text.strip()

# read the input CSV file
cwd = os.getcwd()
df = pd.read_csv(f'{cwd}/data/data/public_datasets/unprocessed_tweets.csv')

# extract hashtags from content
df['tags'] = df['content'].apply(extract_hashtags)

# remove rows with no hashtags
df = df.dropna(subset=['tags'])

# clean the text by removing hashtags
df['text'] = df.apply(lambda row: clean_text(row['content'], row['tags']), axis=1)

# select only the required columns
result_df = df[['text', 'tags']]

result_df.to_csv(f'{cwd}/data/data/public_datasets/processed_tweets.csv', index=False) 