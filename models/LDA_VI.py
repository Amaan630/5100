import random
from collections import Counter
import nltk
import matplotlib.pyplot as plt
import re
from re import RegexFlag
from wordcloud import WordCloud
import sys
import numpy as np
sys.path.append('../preprocess')
from preprocess import load_data
import seaborn as sns
from scipy.special import gammaln, digamma
from nltk.corpus import stopwords
import pandas as pd
from itertools import product
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


class LDA_VB:
    def __init__(self, num_topics, alpha=0.05, beta=0.01, max_iter=100, tolerance=1e-5):
        self.K = num_topics  # number of topics
        self.alpha = alpha  # Dirichlet prior on document-topic distributions
        self.beta = beta  # Dirichlet prior on topic-word distributions
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, docs, vocab):
        # Initialize variables
        self.D = len(docs)  # number of documents
        self.V = len(vocab)  # vocabulary size
        self.vocab = vocab

        # Initialize variational parameters
        # phi: document-word-topic distribution (D x N x K)
        self.phi = [np.random.dirichlet(np.ones(self.K), size=len(doc)) for doc in docs]
        # gamma: document-topic distribution (D x K)
        self.gamma = np.random.gamma(100., 1. / 100., (self.D, self.K))
        # beta: topic-word distribution (K x V)
        self.lambda_param = np.random.gamma(100., 1. / 100., (self.K, self.V))

        # Convert documents to word indices
        self.word_ids = [[vocab.index(word) for word in doc] for doc in docs]

        # Run variational EM
        old_likelihood = -np.inf
        for iteration in range(self.max_iter):
            # E-step
            self._e_step(docs)

            # M-step
            self._m_step(docs)

            # Compute likelihood
            likelihood = self._compute_likelihood(docs)
            print(f"Iteration {iteration}: likelihood = {likelihood}")

            # Check convergence
            if abs(likelihood - old_likelihood) < self.tolerance:
                break
            old_likelihood = likelihood

    def _e_step(self, docs):
        # Update phi and gamma for each document
        for d, doc in enumerate(docs):
            # Initialize gamma for this document
            self.gamma[d] = self.alpha + np.zeros(self.K)

            # Update phi and gamma until convergence
            for _ in range(20):  # max inner iterations
                # Update phi
                for n, word_id in enumerate(self.word_ids[d]):
                    log_phi = np.log(self.lambda_param[:, word_id]) + \
                              digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d]))
                    log_phi = log_phi - np.max(log_phi)  # numerical stability
                    self.phi[d][n] = np.exp(log_phi)
                    self.phi[d][n] = self.phi[d][n] / np.sum(self.phi[d][n])

                # Update gamma
                self.gamma[d] = self.alpha + np.sum(self.phi[d], axis=0)

    def _m_step(self, docs):
        # Update lambda (topic-word distribution)
        self.lambda_param = np.zeros((self.K, self.V)) + self.beta
        for d, doc in enumerate(docs):
            for n, word_id in enumerate(self.word_ids[d]):
                self.lambda_param[:, word_id] += self.phi[d][n]

    def _compute_likelihood(self, docs):
        likelihood = 0

        # Add E[log p(theta | alpha) - log q(theta | gamma)]
        likelihood += np.sum(
            (self.alpha - self.gamma) * digamma(self.gamma) -
            gammaln(self.alpha) + gammaln(self.gamma)
        )

        # Add E[log p(z | theta) - log q(z | phi)]
        for d, doc in enumerate(docs):
            digamma_gamma = digamma(self.gamma[d])  # Calculate once
            digamma_sum = digamma(np.sum(self.gamma[d]))  # Calculate once
            likelihood += np.sum(
                self.phi[d] * (digamma_gamma - digamma_sum)
            )
            likelihood -= np.sum(self.phi[d] * np.log(self.phi[d] + 1e-100))

        return likelihood

    def get_topic_words(self, n_words=10):
        # Get the top words for each topic
        topic_words = []
        for k in range(self.K):
            top_words = np.argsort(self.lambda_param[k])[-n_words:][::-1]
            topic_words.append([self.vocab[i] for i in top_words])
        return topic_words

    def plot_words_clouds_topic(self, topic_names, plt):
        for topic in range(self.K):
            # Get word probabilities for this topic
            word_probs = {self.vocab[i]: self.lambda_param[topic, i]
                          for i in range(self.V)}

            print(f"\nAnalyzing Topic '{topic_names[topic]}':")
            print("Top words in this topic:")

            # Sort words by probability
            sorted_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)
            for word, prob in sorted_words[:10]:
                print(f"  {word}: {prob:.4f}")

            if len(sorted_words) == 0:
                print(f"WARNING: Topic '{topic_names[topic]}' has no words!")
                continue

            # Create text for wordcloud
            wordcloud = WordCloud().generate_from_frequencies(word_probs)
            plt.figure()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("Topic #" + str(topic_names[topic]))
            plt.show()




input_file_path = "../data/preprocessed/combined_preprocessed_denoised.csv"
output_file_path = "../data/processed/lda_processed_round_1.csv"

# Read the CSV file into a pandas DataFrame
df_cleaned = load_data(input_file_path)

topics = df_cleaned['topic'].unique().tolist()
random.shuffle(topics)

output_df = pd.DataFrame(columns=['lda_processed_text'])


def pre_process_documents(documents):
    stop_words = set(stopwords.words('english'))
    processed_docs = []

    for doc in documents:
        if not isinstance(doc[0], str):  # Skip non-string entries
            continue
        # Convert to lowercase and split into words
        words = doc[0].split()
        # Remove special characters and numbers
        words = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in words]
        # Remove empty strings and stop words
        words = [word for word in words if word and word not in stop_words]
        processed_docs.append(words)

    return processed_docs


batch_size = 4  # 4 tweets per topic
iterations = 250  # Number of iterations
num_topics = 4
max_iteration = 10
coherence_scores = []
# Perform batch training
for iteration in range(iterations):
    documents = []
    selected_topics = set()  # Keep track of selected topics

    while len(selected_topics) < num_topics:
        # Randomly select a topic not already chosen
        available_topics = list(set(topics) - selected_topics)
        if not available_topics:
            print(f"Not enough distinct topics available for iteration {iteration}")
            break
        topic = random.choice(available_topics)

        # Extract tweets for the topic
        topic_tweets = df_cleaned[df_cleaned['topic'] == topic]

        # Select tweets for this topic without the 'used_indices' condition
        selected_indices = random.sample(topic_tweets.index.tolist(), batch_size)
        tweets = topic_tweets.loc[selected_indices, 'processed_text'].tolist()
        documents.extend([[tweet] for tweet in tweets])
        selected_topics.add(topic)

    # If not enough topics were selected, skip this iteration
    if len(selected_topics) < num_topics:
        print(f"Skipping iteration {iteration} due to insufficient topics.")
        continue

    processed_docs = pre_process_documents(documents)
    dictionary = Dictionary(processed_docs)
    vocab = sorted(list(set(word for doc in processed_docs for word in doc)))

    alpha_values = [0.01, 0.1, 1.0, 5]
    beta_values = [0.01, 0.5, 1.0]

    best_coherence = -float('inf')
    best_params = None
    best_topic_words = None

    for alpha, beta in product(alpha_values, beta_values):
        # Train the model with current alpha and beta
        lda = LDA_VB(num_topics=num_topics, alpha=alpha, beta=beta, max_iter=max_iteration)
        lda.fit(processed_docs, vocab)

        # Compute coherence score
        topic_words = lda.get_topic_words(n_words=10)
        topics_iteration = [[word for word in topic] for topic in topic_words]
        coherence_model = CoherenceModel(topics=topics_iteration, texts=processed_docs, dictionary=dictionary,
                                         coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        # Update best params if current coherence is higher
        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_params = (alpha, beta)
            best_topic_words = topic_words

    topic_strings = [' '.join(words) for words in best_topic_words]
    temp_df = pd.DataFrame({'lda_processed_text': topic_strings})

    # Concatenate the new rows to the output DataFrame
    output_df = pd.concat([output_df, temp_df], ignore_index=True)

    coherence_scores.append(best_coherence)  # Store the coherence score for this iteration
    print(f"Iteration {iteration}: Coherence Score = {best_coherence}")

# Calculate and print the average coherence score
if coherence_scores:
    avg_coherence_score = sum(coherence_scores) / len(coherence_scores)
    print(f"\nAverage Coherence Score across all iterations: {avg_coherence_score}")
else:
    print("No coherence scores were calculated.")

output_df.to_csv(output_file_path, index=False)

print(f"LDA batch processing completed. Results saved to {output_file_path}")# Save the output DataFrame to a CSV file
