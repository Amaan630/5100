# Social media Topic modeling and Hashtag generation using LDA

This project aims to analyze a collection of tweets to uncover underlying topics, trends, and sentiments. Social media platforms like Twitter contain vast amounts of unstructured text data that can provide valuable insights into public opinion, trending topics, and sentiment distribution. We employ the Latent Dirichlet Allocation (LDA) model to extract latent topics within tweets, followed by various post-processing analyses to provide a comprehensive view of the data. This pipeline allows us to transform raw tweets into actionable insights through structured data analysis and visualization.

## Problem Statement

The primary goal is to answer questions such as:
 - What are people talking about on social media? - Extract main topics in the data.
 - How do topics vary over time? - Understand trends and the evolution of topics.
 - What are people’s sentiments around these topics? - Perform sentiment analysis to gauge the positivity, negativity, or neutrality of 
opinions.
 - How can we automate insights into hate speech or hashtags? - Identify specific instances of hate speech and generate relevant hashtags.

## Flowchart
![Alt text](relative%20path/to/img.jpg?raw=true "Title")

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Workflow](#workflow)
  - [1. Tweet Extraction](#1-tweet-extraction)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Data Preparation](#3-data-preparation)
  - [4. Training the LDA Model](#4-training-the-lda-model)
  - [5. Model Testing](#5-model-testing)
  - [6. Post-processing](#6-post-processing)
  - [7. Visualization](#7-visualization)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to:
- Extract topics from tweets using the LDA model.
- Perform sentiment analysis and other post-processing tasks to gain insights into tweet content.
- Visualize results using a graphical user interface.
