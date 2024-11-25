import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from itertools import combinations
import networkx as nx


def dataset_overview(df):
    print("Dataset Overview:")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSample Rows:\n", df.head())


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

    # Bigram and trigram analysis
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
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().sum(axis=0)))

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
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Text Data')
    plt.show()

def sentiment_analysis(df, text_column):
    def get_sentiment(text):
        analysis = TextBlob(text)
        return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

    df['sentiment'] = df[text_column].apply(get_sentiment)
    sentiment_counts = df['sentiment'].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.show()

def hashtag_analysis(df, hashtag_column):
    hashtag_counts = Counter(" ".join(df[hashtag_column].dropna()).split())
    common_hashtags = hashtag_counts.most_common(20)
    print("Top 20 Hashtags:", common_hashtags)

    hashtags, counts = zip(*common_hashtags)
    plt.figure(figsize=(10, 5))
    plt.bar(hashtags, counts)
    plt.title("Top 20 Hashtags")
    plt.xticks(rotation=45)
    plt.show()

def hashtag_co_occurrence(df, hashtag_column):
    co_occurrence = Counter()
    for hashtags in df[hashtag_column].dropna():
        tags = hashtags.split()
        co_occurrence.update(combinations(tags, 2))

    # Create and plot a network graph
    G = nx.Graph()
    for (tag1, tag2), count in co_occurrence.items():
        G.add_edge(tag1, tag2, weight=count)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw_networkx(G, pos, with_labels=True, node_color='skyblue', edge_color='gray')
    plt.title("Hashtag Co-occurrence Network")
    plt.show()

def text_complexity_analysis(df, text_column):
    def vocab_diversity(text):
        words = text.split()
        return len(set(words)) / len(words) if len(words) > 0 else 0

    df['vocab_diversity'] = df[text_column].apply(vocab_diversity)
    plt.figure(figsize=(8, 5))
    plt.hist(df['vocab_diversity'], bins=30, color='purple', edgecolor='black')
    plt.title("Vocabulary Diversity Distribution")
    plt.xlabel("Diversity Score")
    plt.ylabel("Frequency")
    plt.show()


def perform_eda(preprocessed_file, text_column,hashtag_column):
    """Perform all EDA on the preprocessed file."""
    df = pd.read_csv(preprocessed_file)

    print("Overview of the dataset")
    dataset_overview(df)

    print("Performing Word Frequency Analysis...")
    word_frequency_analysis(df, text_column)

    print("Performing Bigram and Trigram Analysis...")
    bigram_trigram_analysis(df, text_column)

    print("Performing Document Length Analysis...")
    document_length_analysis(df, text_column)

    print("Performing TF-IDF Analysis...")
    tfidf_analysis(df, text_column)

    print("Sentiment Analysis...")
    sentiment_analysis(df, text_column)

    print("Text Complexity Analysis...")
    text_complexity_analysis(df, text_column)

    print("Generating Word Cloud...")
    generate_wordcloud(df, text_column)

    print("Hashtags Analysis....")
    hashtag_analysis(df, hashtag_column)

    print("Hashtag Co Occurrence Analysis....")
    hashtag_co_occurrence(df, hashtag_column)
