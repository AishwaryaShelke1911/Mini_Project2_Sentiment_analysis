📌 Introduction

This project builds upon the basics of NLP covered in Mini Project 1 and explores advanced word embeddings techniques such as Word2Vec and GloVe.
These models help transform words into dense vector representations that capture semantic meaning and context, enabling better performance in sentiment analysis tasks.

🏢 Business Context

Just like in Project 1, e-commerce platforms need to analyze customer reviews efficiently. However, instead of Bag of Words, which ignores context, this project leverages vector embeddings to improve sentiment classification.

Key risks of ignoring sentiment:

🚶 Customer Churn

⚠️ Reputation Damage

💸 Financial Loss

Thus, embedding-based sentiment analysis provides deeper insights into customer feedback.

🎯 Problem Statement

The company needs an AI-driven sentiment classification model but with more sophisticated text representation.
Your task: implement Word2Vec (CBOW & Skip-Gram) and GloVe embeddings for vectorization of text data, and use them for sentiment classification.

📊 Dataset

Product ID → Unique identifier for each product

Product Review → Customer feedback text

Sentiment → Label (Positive, Negative, Neutral)

⚙️ Tech Stack & Libraries

Python

Libraries:

pandas, numpy → Data manipulation

matplotlib, seaborn → Visualization

nltk → Preprocessing (stopwords, stemming, lemmatization)

scikit-learn → Random Forest Classifier, train-test split, metrics

gensim → Word2Vec (CBOW, Skip-Gram), GloVe integration

re → Regular expressions

🧹 Text Preprocessing

Removing special characters

Converting to lowercase

Removing extra whitespaces

Removing stopwords

Lemmatization (WordNetLemmatizer)

🔤 Word Embeddings

Word2Vec

CBOW → Predicts target word from context

Skip-Gram → Predicts context words from target

GloVe

Pre-trained vectors (glove.6B.100d) loaded via Gensim

## Pre-trained Embeddings
This project uses [GloVe embeddings](https://nlp.stanford.edu/projects/glove/).  
The file `glove.6B.100d.txt` (~350 MB) is **not included in this repository** due to size limits.  

To download it automatically, just run the notebook/script and it will fetch the embeddings from Stanford’s site.  
Alternatively, you can manually download from [GloVe official page](https://nlp.stanford.edu/projects/glove/).


🤖 Model

Sentence vectors created by averaging word embeddings

Random Forest Classifier trained on embeddings

Evaluation metrics:

Accuracy

📈 Results

Learned dense vector representations for words using Word2Vec & GloVe

Built sentence-level vectors for sentiment classification

Demonstrated how embeddings outperform traditional BoW for capturing meaning

Classification Report

Confusion Matrix
