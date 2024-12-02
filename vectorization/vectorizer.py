import re
import os

from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
from math import log

import pickle
import pandas as pd
import numpy as np


SIMILAR_DOCS_TO_CONSIDER = 1  # number of similar documents to pull hashtags from
HASHTAGS_TO_RETURN = 5
DATA_FILES_TO_USE = ["data/data/processed/lda_processed_round_1.csv"]


class Vectorizer:

    #
    # the vectorizer implements a bag-of-words (bow) approach with bm25 weighting, designed
    # for matching documents where word order is not significant.
    #
    # key features:
    #
    # => bag-of-words: treats each document as an unordered collection of words, which is
    #    appropriate for our use case where input data contains randomly ordered words.
    #
    # => bm25 weighting: uses the bm25 algorithm to compute word importance relative to the
    #    document collection. this helps identify significant words while reducing the impact
    #    of common terms.
    #
    # => normalized vectors: final document vectors are normalized to ensure consistent
    #    similarity computations regardless of document length.
    #

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.vocabulary = {}  # word -> index mapping
        self.idf = None  # inverse document frequency
        self.doc_vectors = None  # document vectors
        self.k1 = k1  # bm25 parameter
        self.b = b  # bm25 parameter
        self.avg_doc_length = 0

    def _preprocess(self, text: str) -> List[str]:
        # convert text to lowercase and remove special chars
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def fit(self, documents: List[str]):
        # build vocabulary and compute idf with bm25 weighting
        word_doc_count = Counter()
        all_words = set()
        doc_lengths = []

        for doc in documents:
            tokens = self._preprocess(doc)
            doc_lengths.append(len(tokens))
            words = set(tokens)
            word_doc_count.update(words)
            all_words.update(words)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.avg_doc_length = np.mean(doc_lengths)

        # compute bm25 idf
        num_docs = len(documents)
        self.idf = np.zeros(len(self.vocabulary))
        for word, count in word_doc_count.items():
            if word in self.vocabulary:
                self.idf[self.vocabulary[word]] = log(
                    (num_docs - count + 0.5) / (count + 0.5) + 1.0
                )

    def transform(self, documents: List[str]) -> np.ndarray:
        if not self.vocabulary or self.idf is None:
            raise ValueError("vectorizer must be fit before transform")

        vectors = np.zeros((len(documents), len(self.vocabulary)))

        for doc_idx, doc in enumerate(documents):
            tokens = self._preprocess(doc)
            doc_length = len(tokens)
            word_counts = Counter(tokens)

            for word, count in word_counts.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    # compute bm25 term frequency
                    tf = (count * (self.k1 + 1)) / (
                        count
                        + self.k1
                        * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                    )
                    vectors[doc_idx, word_idx] = tf * self.idf[word_idx]

        # normalize vectors
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1
        vectors = vectors / norms[:, np.newaxis]

        return vectors


class HashtagRecommender:
    def __init__(self):
        self.vectorizer = Vectorizer()
        self.vectors = None
        self.texts = None
        self.vector_cache_path = "cache/vectors.pkl"
        self.hashtag_cache_path = "cache/hashtags.pkl"
        self.data_files = DATA_FILES_TO_USE

        # create cache directory if it doesn't exist
        os.makedirs("cache", exist_ok=True)

    def _load_and_process_csvs(self) -> None:
        """Load specified CSV files and process them"""
        all_texts = []

        for csv_file in self.data_files:
            df = pd.read_csv(csv_file)
            if "lda_processed_text" in df.columns:
                all_texts.extend(df["lda_processed_text"].tolist())

        if not all_texts:
            raise ValueError(
                "No valid data found in CSV files. Please check that your CSV files contain 'lda_processed_text' column."
            )

        self.vectorizer.fit(all_texts)
        self.vectors = self.vectorizer.transform(all_texts)
        self.texts = all_texts

        with open(self.vector_cache_path, "wb") as f:
            pickle.dump((self.vectorizer, self.vectors), f)
        with open(self.hashtag_cache_path, "wb") as f:
            pickle.dump(self.texts, f)

    def initialize(self) -> None:
        """Initialize or load from cache"""
        if os.path.exists(self.vector_cache_path) and os.path.exists(
            self.hashtag_cache_path
        ):
            with open(self.vector_cache_path, "rb") as f:
                self.vectorizer, self.vectors = pickle.load(f)
            with open(self.hashtag_cache_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self._load_and_process_csvs()

    def get_top_hashtags(
        self, text: str, n: int = HASHTAGS_TO_RETURN
    ) -> Dict[str, float]:
        """Get top n hashtags for input text"""
        if self.vectors is None or not self.texts:
            raise ValueError("Please initialize the recommender first")

        query_vector = self.vectorizer.transform([text])

        # get similarities and find most similar document
        similarities = np.dot(self.vectors, query_vector.T).flatten()
        top_indices = np.argsort(similarities)[-SIMILAR_DOCS_TO_CONSIDER:][::-1]

        # get random words from the most similar documents
        hashtag_counts = {}
        for idx in top_indices:
            if idx < len(self.texts):
                words = self.texts[idx].split()
                selected_words = np.random.choice(
                    words, size=min(n, len(words)), replace=False
                )
                for word in selected_words:
                    tag = f"#{word}" if not word.startswith("#") else word
                    hashtag_counts[tag] = similarities[idx]

        if not hashtag_counts:
            return {}

        # return top n hashtags
        sorted_hashtags = sorted(
            hashtag_counts.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_hashtags[:n])


if __name__ == "__main__":
    print("Initializing hashtag recommender...")
    recommender = HashtagRecommender()
    recommender.initialize()
    print("Ready to recommend hashtags!")
    print("\nEnter text to get hashtag recommendations (or 'quit' to exit)")

    while True:
        print("\n> ", end="")
        text = input()

        if not text.strip():
            continue

        try:
            top_hashtags = recommender.get_top_hashtags(text)
            print("\nRecommended hashtags:")
            for tag, score in top_hashtags.items():
                print(f"  {tag}")
        except Exception as e:
            print(f"Error: {str(e)}")
