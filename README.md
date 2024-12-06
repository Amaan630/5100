# Social media Topic modeling and Hashtag generation using LDA

This project aims to analyze a collection of tweets to uncover underlying topics, trends, and sentiments. Social media platforms like Twitter contain vast amounts of unstructured text data that can provide valuable insights into public opinion, trending topics, and sentiment distribution. We employ the Latent Dirichlet Allocation (LDA) model to extract latent topics within tweets, followed by various post-processing analyses to provide a comprehensive view of the data. This pipeline allows us to transform raw tweets into actionable insights through structured data analysis and visualization.

## Problem Statement

The primary goal is to answer questions such as:
 - What are people talking about on social media? - Extract main topics in the data.
 - How do topics vary over time? - Understand trends and the evolution of topics.
 - What are people’s sentiments around these topics? - Perform sentiment analysis to gauge the positivity, negativity, or neutrality of 
opinions.
 - How can we automate insights into hate speech or hashtags? - Identify specific instances of hate speech and generate relevant hashtags.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open the browser and navigate to http://127.0.0.1:5000/ to see the web app.

## Flowchart
![Alt text](HighLevelDiagram.png?raw=true "Title")

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Workflow](#workflow)
  - [1. Tweet Extraction](#1-tweet-extraction)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Data Preparation](#3-data-preparation)
  - [4. Training the LDA Model](#4-training-the-lda-model)
  - [5. Model Testing](#5-model-testing)
  - [6. Visualization](#7-visualization)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to:
- Extract topics from tweets using the LDA model.
- Perform sentiment analysis and other post-processing tasks to gain insights into tweet content.
- Visualize results using a graphical user interface.

## Folder Structure

```
data/  # contains datasets, both preprocessed and raw
    preprocessed/
    processed/
    dataset.json  # raw dataset obtained from https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi/blob/main/dataset/split_random/train_random.multi.json
    process_tweets.py  # script to process the raw dataset into csv files
    query_rows.py  # script to get the number of rows in each csv file in the data directory. it is just for reference.
    *.csv  # raw csv datasets generated from the process_tweets.py script

models/  # contains the trained LDA model

preprocess/  # contains the logic to preprocess the data
    preprocess.py #Preprocess the dataset and create a new csv file with the preprocessed text
    denoising.py #Remove additional noise from the data
    eda.py #Performing the Exploratory Data Analysis to further understand the data

static/  # contains static files for the web app (styling)

templates/  # contains html templates for the web app

vectorization/  # contains the logic to vectorize the data

app.py  # flask app to serve the web app
```

## Workflow

### 1. Tweet Extraction
- Data is sourced from the Cardiff NLP tweet topic dataset
- Dataset contains labeled tweets with multiple topics
- Raw data is stored in JSON format in the data directory
- Each tweet includes date, text, ID, and topic labels

### 2. Data Preprocessing
- Text cleaning and normalization
- Removal of special characters, URLs, and mentions
- Tokenization and stop word removal
- Part-of-speech tagging to extract relevant nouns
- Denoising of text using custom preprocessing pipeline

### 3. Data Preparation
- Conversion of raw JSON to structured CSV format
- Organization of data into preprocessed and processed directories
- Creation of vocabulary and word indices for LDA model
- Batch processing of tweets for efficient training

### 4. Training the LDA Model
- Implementation of Latent Dirichlet Allocation using Variational Bayes
- Hyperparameter tuning (alpha, beta)
- Topic modeling with configurable number of topics
- Iterative training process with convergence monitoring
- Generation of topic-word distributions

### 5. Model Testing
- Evaluation of topic coherence
- Assessment of model convergence

### 6. Visualization
- Interactive web interface using Flask


## Setup and Installation

1. Clone the repository:

```bash
git clone git@github.com:Amaan630/5100.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the Flask application:

```bash
python app.py
```


2. Access the web interface:
- Open browser and navigate to http://127.0.0.1:5000/
- Use the interface to:
  - Input tweets for topic analysis
  - Generate hashtags

## Contributing

1. Clone the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
