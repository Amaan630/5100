import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary nltk data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def load_data(file_path):
  """Load the dataset."""
  return pd.read_csv(file_path)


def clean_text(text):
  """Clean text by removing URLs, mentions, hashtags, punctuation, and stop words, and apply lemmatization."""
  text = text.lower()  # Convert to lowercase
  text = re.sub(r"http\S+|www\S+|https\S+", '', text,
                flags=re.MULTILINE)  # Remove URLs
  text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
  text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
  text = " ".join([word for word in text.split() if
                   word not in stop_words])  # Remove stop words
  text = " ".join(
      [lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatize
  return text


def preprocess_data(df, text_column):
  df['cleaned_text'] = df[text_column].apply(clean_text)
  return df


# Pre-processing Analysis Functions

def word_frequency_analysis(df, column, top_n=20):
  """Plot the most common words in the cleaned text data."""
  all_words = " ".join(df[column]).split()
  word_counts = Counter(all_words)
  common_words = word_counts.most_common(top_n)

  words, counts = zip(*common_words)
  plt.figure(figsize=(10, 5))
  plt.bar(words, counts)
  plt.title(f'Top {top_n} Most Common Words')
  plt.xticks(rotation=45)
  plt.show()


def bigram_trigram_analysis(df, column, top_n=10):
  """Analyze and display the most common bigrams and trigrams."""
  all_words = [text.split() for text in df[column]]

  # Bigram analysis
  bigrams = Counter()
  trigrams = Counter()
  for words in all_words:
    bigrams.update(zip(words, words[1:]))
    trigrams.update(zip(words, words[1:], words[2:]))

  # Display top bigrams
  top_bigrams = bigrams.most_common(top_n)
  bigram_labels, bigram_counts = zip(*top_bigrams)
  bigram_labels = [" ".join(bigram) for bigram in bigram_labels]

  plt.figure(figsize=(10, 5))
  plt.bar(bigram_labels, bigram_counts)
  plt.title(f'Top {top_n} Most Common Bigrams')
  plt.xticks(rotation=45)
  plt.show()

  # Display top trigrams
  top_trigrams = trigrams.most_common(top_n)
  trigram_labels, trigram_counts = zip(*top_trigrams)
  trigram_labels = [" ".join(trigram) for trigram in trigram_labels]

  plt.figure(figsize=(10, 5))
  plt.bar(trigram_labels, trigram_counts)
  plt.title(f'Top {top_n} Most Common Trigrams')
  plt.xticks(rotation=45)
  plt.show()


def document_length_analysis(df, column):
  """Plot the distribution of document lengths."""
  df['doc_length'] = df[column].apply(lambda x: len(x.split()))
  plt.figure(figsize=(10, 5))
  plt.hist(df['doc_length'], bins=30, color='skyblue', edgecolor='black')
  plt.title('Document Length Distribution')
  plt.xlabel('Number of Words')
  plt.ylabel('Frequency')
  plt.show()


def tfidf_analysis(df, column, top_n=20):
  """Perform initial TF-IDF analysis to find high-value terms."""
  vectorizer = TfidfVectorizer(max_features=top_n)
  tfidf_matrix = vectorizer.fit_transform(df[column])
  tfidf_scores = dict(
    zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().sum(axis=0)))

  # Sort terms by score
  sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
  terms, scores = zip(*sorted_tfidf)

  plt.figure(figsize=(10, 5))
  plt.bar(terms, scores)
  plt.title(f'Top {top_n} TF-IDF Terms')
  plt.xticks(rotation=45)
  plt.show()


def generate_wordcloud(df, column):
  """Generate a word cloud for the cleaned text."""
  text = " ".join(df[column])
  wordcloud = WordCloud(width=800, height=400,
                        background_color="white").generate(text)

  plt.figure(figsize=(10, 5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.title('Word Cloud of Text Data')
  plt.show()