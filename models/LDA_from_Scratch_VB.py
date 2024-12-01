import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import nltk
from scipy.special import gammaln, digamma
nltk.download('stopwords')

class LDA_VB:
    def __init__(self, num_topics, alpha=0.1, beta=0.1, max_iter=100, tolerance=1e-4):
        self.K = num_topics  # number of topics
        self.alpha = alpha   # Dirichlet prior on document-topic distributions
        self.beta = beta     # Dirichlet prior on topic-word distributions
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
        self.gamma = np.random.gamma(100., 1./100., (self.D, self.K))
        # beta: topic-word distribution (K x V)
        self.lambda_param = np.random.gamma(100., 1./100., (self.K, self.V))
        
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

def pre_process_documents(documents):
    stop_words = set(stopwords.words('english'))
    processed_docs = []
    
    for doc in documents:
        # Convert to lowercase and split into words
        words = doc[0].lower().split()
        # Remove special characters and numbers
        words = [re.sub(r'[^a-zA-Z]', '', word) for word in words]
        # Remove empty strings and stop words
        words = [word for word in words if word and word not in stop_words]
        processed_docs.append(words)
    
    return processed_docs

# Example usage
documents = [
    # Sports (3 documents)
    ["Players practice soccer on the field."],
    ["Basketball players shoot hoops."],
    ["The tennis match was intense."],
    
    # Animals (3 documents)
    ["The brown fox and the red squirrel played in the forest together."],
    ["Birds chirp while cats and dogs chase each other in the garden."],
    ["Rabbits hop around while deer graze peacefully in the meadow."],
    
    # Food (3 documents)
    ["The chef prepared pasta with fresh tomatoes and aromatic herbs."],
    ["Breakfast included eggs, bacon, toast, and fresh orange juice."],
    ["The cake was decorated with chocolate, cream, and fresh berries."],
    
    # Weather (3 documents)
    ["The sky is clear blue today with gentle breezes blowing."],
    ["Dark clouds gathered as thunder rumbled in the distance."],
    ["Snow fell softly, covering the ground in a white blanket."]
]

topic_names = ['animals', 'sports', 'weather', 'food']

# Preprocess documents
processed_docs = pre_process_documents(documents)

# Create vocabulary
vocab = sorted(list(set(word for doc in processed_docs for word in doc)))

# Initialize and fit LDA model
lda = LDA_VB(num_topics=len(topic_names))
lda.fit(processed_docs, vocab)

# Plot word clouds
lda.plot_words_clouds_topic(topic_names, plt)

# Print top words for each topic
print("\nTop words per topic:")
topic_words = lda.get_topic_words()
for topic_idx, words in enumerate(topic_words):
    print(f"\nTopic {topic_names[topic_idx]}: {', '.join(words)}")

