#!/usr/bin/env python
# coding: utf-8

# ## Overview
# Project taken from Google Tensorboard examples. What was used in that example did not perform any stemming or lemmetization of the corpus. This is my version where I'll try to improve the results thru tokenization, lemmetization and encoding. Ultimately I want to use my own corpus.

# https://distill.pub/2016/misread-tsne/

# ## Prepare Enviornment

# In[63]:


get_ipython().system('pip install nltk')

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins import projector
import csv
import pandas as pd
import tensorflow as tf

# NLP libraries and methods
import nltk
#nltk.download("all")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk import FreqDist

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from collections import Counter
from google.colab import drive


# In[64]:


# This seems to propagate credentials better from its own cell

from google.colab import drive
drive.mount('/content/drive')


# # IMDB Data

# Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.
# For more dataset information, please go through the following link,
# http://ai.stanford.edu/~amaas/data/sentiment/

# ## Data Loading and Preprocessing

# In[65]:


# Access tensorflow dataset
# tfds is preprocessed, but I want to start with raw data
# allowing me to improve the Lemmetization

data = tfds.load("imdb_reviews/plain_text", as_supervised=True)
full_dataset = data['train'].concatenate(data['test'])

# Convert tfds data to csv
# then into a pandas df

with open('imdb_reviews.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['review', 'sentiment'])  # Writing header

    for review, label in tfds.as_numpy(full_dataset):
        writer.writerow([review.decode('utf-8'), label])

# Load csv into a DataFrame
file_path = 'imdb_reviews.csv'
df_raw = pd.read_csv(file_path)


# The pre-processed form of this data already has sentiment scores.
# Im going to run my own sentimization on this more raw for of the data.

def sentimentization(df):
    stm = SentimentIntensityAnalyzer()  # Instantiate the analyzer
    sentiment_score = []
    sentiment_class = []

    # Iterate over each review in the DataFrame
    for review in df['review']:
        StmScore = stm.polarity_scores(review)
        # Apply a sentiment classification based on the sentiment score
        if StmScore['compound'] > 0.7:
            sentiment_class.append('positive')
        elif StmScore['compound'] < -0.7:
            sentiment_class.append('negative')
        else:
            sentiment_class.append('neutral')

        sentiment_score.append(StmScore['compound'])

    # Add the lists as new columns in the DataFrame
    df['sentiment_score'] = sentiment_score
    df['sentiment_class'] = sentiment_class

    return df


# Tokenize each review with NLTK prior to lemmetization
# The resuling stings will be re-encoded with Tensorflow later on

def tokenize(df):
    # Initialize an empty list to store tokenized reviews
    tokenized_reviews = []

    # Iterate over each review in the DataFrame
    for review in df['review']:

    # Tokenize the review, convert to lower case
        tokens = word_tokenize(review.lower())

    # Filter out small character words as they are unlikely to carry much payload
        filtered_tokens = [token for token in tokens if len(token) > 2]

    # Append the filtered tokens to the list
        tokenized_reviews.append(filtered_tokens)

    # Add the list as a new column in the DataFrame
    df['review_token'] = tokenized_reviews

    return df

# Lemmatization of the data

lemmatizer = WordNetLemmatizer()

def lemmy(df):
    # Initialize an empty list to store lemmatized reviews
    lemmatized_reviews = []

    # Iterate over each tokenized review in the DataFrame
    for review in df['review_token']:
        lemmatized_review = [lemmatizer.lemmatize(word) for word in review]

        lemmatized_reviews.append(lemmatized_review)

    # Add the list as a new column in the DataFrame
    df['lemmatized_reviews'] = lemmatized_reviews

    return df


# Filter out stopwords from the NLTK tokenized df

mystopwords = set(stopwords.words("english"))

# Add additional words to stopwords
additional_stopwords = ["movie", "film"]  # As this is a film corpus, these have no apparent value

mystopwords.update(additional_stopwords)

def stop_words(df):
    # Initialize an empty list to store filtered reviews
    reviews_filtered = []

    # Iterate over each tokenized review in the DataFrame
    for review in df['review_token']:
        # Filter out stopwords and non-alphabetical tokens
        filtered_review = [word for word in review if word not in mystopwords and word.isalpha()]
        # Append the filtered review to the list
        reviews_filtered.append(filtered_review)

    # Add the list as a new column in the DataFrame
    df['review_token_nostop'] = reviews_filtered

    return df


df_sent = sentimentization(df_raw)
df_token = tokenize(df_raw)
df_token = df_token.drop(columns=['review'])
df_lem = lemmy(df_token)
df_tok_stop = stop_words(df_lem)


# Flatten the new target

# Use pd.get_dummies to one-hot encode the 'sentiment_class' column
df_tok_stop = pd.get_dummies(df_tok_stop, columns=['sentiment_class'], prefix="sentiment_class")

# Drop the unecessary columns

df_tok_stop.drop(['sentiment','sentiment_score','review_token', 'lemmatized_reviews', 'sentiment_class_negative', 'sentiment_class_neutral'], axis=1, inplace=True)

df_tok_stop.head()


# ## Modeling with TensorFlow

# In[66]:


from collections import Counter

# Combine all words from lemmatized reviews into a single list
all_words = []
for reviews in df_tok_stop['review_token_nostop']:
    all_words.extend(reviews)

# Create a Counter object to count word frequencies
word_counts = Counter(all_words)

# Calculate the total unique word count
total_unique_words = len(word_counts)

# Calculate the total word count
total_word_count = sum(word_counts.values())

# Calculate the target frequency for 80% of unique words
target_frequency = 0.8 * total_word_count

# Sort the word frequencies in descending order
sorted_word_counts = sorted(word_counts.values(), reverse=True)

# Calculate the number of words required to reach 80% of unique words
num_words_for_80_percent = 0
cumulative_frequency = 0

for frequency in sorted_word_counts:
    cumulative_frequency += frequency
    num_words_for_80_percent += 1
    if cumulative_frequency >= target_frequency:
        break

# Create a DataFrame from word frequencies
#word_freq_df = pd.DataFrame({'Word': word_counts.keys(), 'Frequency': word_counts.values()})

# Export the DataFrame to a CSV file
#word_freq_df.to_csv('word_frequencies.csv', index=False)

# Print the results
print(f"Total Unique Word Count: {total_unique_words}")
print(f"Total Word Count: {total_word_count}")
print(f"Number of Words for 80% of Unique Words: {num_words_for_80_percent}")


# In[67]:


# This is the tensorflow tokenization
# Tokenize and model have unique names to address possible polution
# with subsequent that use the same code strucuture but different files

# Initialize the tokenizer
tokenizer_imdb = Tokenizer(num_words=4384)
tokenizer_imdb.fit_on_texts(df_tok_stop['review_token_nostop'])

# Convert texts to sequences
sequences = tokenizer_imdb.texts_to_sequences(df_tok_stop['review_token_nostop'])

# Padding sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=200)

# Assumes I have already tokenized and padded x-values into `padded_sequences`

X = padded_sequences
y = df_tok_stop['sentiment_class_positive'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define your model architecture
vocab_size_imdb = 4384  # Replace with the actual size of your vocabulary
embedding_dim_imdb = 16  # The size of the embedding vectors

model_imdb = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size_imdb, embedding_dim_imdb, input_length=X_train.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# Compile the model
model_imdb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_imdb.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# ## TensorBoard Visualization:

# In[68]:


# Create a one-dimensional metadata file

# Specify the log directory and metadata file path
log_dir = '/content/drive/MyDrive/imdb/tsnse/logs'
metadata_file_path = os.path.join(log_dir, 'metadata.tsv')

# Ensure the directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

word_index = tokenizer_imdb.word_index

# Number of words in the word index
word_index_length = len(word_index)

# Create the metadata file for TensorBoard
with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
    word_counter = 0
    for word in word_index:
        if word_counter < vocab_size:
            metadata_file.write(word + '\n')
            word_counter += 1


weights = tf.Variable(model_imdb.layers[0].get_weights()[0])
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# TensorBoard Projector Configuration
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
#embedding.metadata_path = '/logs/imdb-example2/metadata.tsv'
projector.visualize_embeddings(log_dir, config)


#'model' is your trained Keras model from above
embedding_layer = model_imdb.layers[0]
embeddings = embedding_layer.get_weights()[0]
num_tensors = embeddings.shape[0]

# Count the number of lines in metadata.tsv
metadata_file_path = '/content/drive/MyDrive/imdb/tsnse/logs/metadata.tsv'
with open(metadata_file_path, 'r', encoding='utf-8') as file:
    num_words = sum(1 for line in file)

# Counts need to agree
print(f"Number of tensors: {num_tensors}")
print(f"Number of words in metadata file: {num_words}")

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir /content/drive/MyDrive/imdb/tsnse/logs/')
#%reload_ext tensorboard


# # E9 Project

# Additional effort to leverage this technolgy in the development of a virtual mechanic for classic cars.

# ## Data Loading and Preprocessing

# In[69]:


# Access dataset
# Data here is from a corpus I compiled from a classic car forum:
# https://e9coupe.com
# The data has already been through lemmetization, but not stop word removal


# Load csv into a DataFrame
#file_path = '/content/df_posts.csv'
df_posts = pd.read_csv('/content/drive/MyDrive/Data_sets/df_posts.csv')

df_posts['post_raw'] = df_posts['post_raw'].apply(lambda x: str(x))


# Tokenization
post_tokens = []

for text in df_posts['post_raw']:
    post_token = word_tokenize(text.lower())  # Tokenize each post
    post_token = [token for token in post_token if len(token) > 1]  # Filter out single-character tokens
    post_tokens.append(post_token)

# Add tokenized posts as a new column in the DataFrame
df_posts['post_tokens'] = post_tokens

# Stop words
mystopwords = set(stopwords.words("english"))
post_filtered = []

for tokens in df_posts['post_tokens']:
    filtered_tokens = [token for token in tokens if token not in mystopwords and token.isalpha()]
    post_filtered.append(filtered_tokens)

df_posts['post_filtered'] = post_filtered  # Add filtered tokens as a new column

# Lemmatization
lemmatizer = WordNetLemmatizer()

lemm_list = []

for words in df_posts['post_filtered']:
    lemm_words = [lemmatizer.lemmatize(word) for word in words]
    lemm_list.append(lemm_words)

# Add lemmatized words as a new column
df_posts['post_lemm'] = lemm_list

df_posts['post_lemm_str'] = df_posts['post_lemm'].apply(lambda x: str(x))


# In[70]:


# Create a sentiment score and classification from each review in the raw df

def sentimentization(df):
    stm = SentimentIntensityAnalyzer()  # Instantiate the analyzer
    sentiment_score = []
    sentiment_class = []

    # Iterate over each review in the DataFrame
    for post in df['post_lemm_str']:
        StmScore = stm.polarity_scores(post)

        # Apply a sentiment classification based on the sentiment score
        if StmScore['compound'] > 0.7:
            sentiment_class.append('positive')
        elif StmScore['compound'] < -0.7:
            sentiment_class.append('negative')
        else:
            sentiment_class.append('neutral')

        sentiment_score.append(StmScore['compound'])

    # Add the lists as new columns in the DataFrame
    df['sentiment_score'] = sentiment_score
    df['sentiment_class'] = sentiment_class

    return df


df_sent = sentimentization(df_posts)


# Flatten the new target

# Use pd.get_dummies to one-hot encode the 'sentiment_class' column
df_sent = pd.get_dummies(df_sent, columns=['sentiment_class'], prefix="sentiment_class")

# Drop the unecessary columns

df_sent.drop(['thread_id','thread_title','post_id','timestamp','post_number', 'post_raw', 'post_tokens', 'post_filtered','post_lemm','sentiment_score'], axis=1, inplace=True)

#df_tok_stop.head()


# In[71]:


from collections import Counter

# Combine all words from lemmatized reviews into a single list
all_words = []
for reviews in df_sent['post_lemm_str']:
    all_words.extend(reviews)

# Create a Counter object to count word frequencies
word_counts = Counter(all_words)

# Calculate the total unique word count
total_unique_words = len(word_counts)

# Calculate the total word count
total_word_count = sum(word_counts.values())

# Calculate the target frequency for 80% of unique words
target_frequency = 0.8 * total_word_count

# Sort the word frequencies in descending order
sorted_word_counts = sorted(word_counts.values(), reverse=True)

# Calculate the number of words required to reach 80% of unique words
num_words_for_80_percent = 0
cumulative_frequency = 0

for frequency in sorted_word_counts:
    cumulative_frequency += frequency
    num_words_for_80_percent += 1
    if cumulative_frequency >= target_frequency:
        break

# Create a DataFrame from word frequencies
#word_freq_df = pd.DataFrame({'Word': word_counts.keys(), 'Frequency': word_counts.values()})

# Export the DataFrame to a CSV file
#word_freq_df.to_csv('word_frequencies.csv', index=False)

# Print the results
print(f"Total Unique Word Count: {total_unique_words}")
print(f"Total Word Count: {total_word_count}")
print(f"Number of Words for 80% of Unique Words: {num_words_for_80_percent}")


# In[72]:


# This is the tensorflow tokenization

# Initialize the tokenizer
tokenizer_e9 = Tokenizer(num_words=1000)
tokenizer_e9.fit_on_texts(df_sent['post_lemm_str'])

# Convert texts to sequences
sequences = tokenizer_e9.texts_to_sequences(df_sent['post_lemm_str'])

# Padding sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=200)

# Assumes I have already tokenized and padded x-values into `padded_sequences`
X = padded_sequences
y = df_sent['sentiment_class_neutral'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define your model architecture
vocab_size_e9 = 1000  # Replace with the actual size of your vocabulary
embedding_dim_e9 = 16  # The size of the embedding vectors

model_e9 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size_e9, embedding_dim_e9, input_length=X_train.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# Compile the model
model_e9.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_e9.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[73]:


# Create a one-dimensional metadata file

# Specify the log directory and metadata file path
#log_dir = '/logs/imdb-example2/'
log_dir = '/content/drive/MyDrive/e9/tsnse/logs'
metadata_file_path = os.path.join(log_dir, 'metadata.tsv')

# Ensure the directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

word_index = tokenizer.word_index

# Number of words in the word index
word_index_length = len(word_index)

# Create the metadata file for TensorBoard
with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
    word_counter = 0
    for word in word_index:
        if word_counter < vocab_size:
            metadata_file.write(word + '\n')
            word_counter += 1


weights = tf.Variable(model_e9.layers[0].get_weights()[0])
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# TensorBoard Projector Configuration
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
#embedding.metadata_path = '/logs/imdb-example2/metadata.tsv'
projector.visualize_embeddings(log_dir, config)


# Assuming 'model' is your trained Keras model
embedding_layer = model_e9.layers[0]  # Adjust the index if your embedding layer is at a different position
embeddings = embedding_layer.get_weights()[0]
num_tensors = embeddings.shape[0]

# Count the number of lines in metadata.tsv
metadata_file_path = '/content/drive/MyDrive/e9/tsnse/logs/metadata.tsv'
with open(metadata_file_path, 'r', encoding='utf-8') as file:
    num_words = sum(1 for line in file)

# Counts need to agree
print(f"Number of tensors: {num_tensors}")
print(f"Number of words in metadata file: {num_words}")

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir /content/drive/MyDrive/e9/tsnse/logs/')
#%reload_ext tensorboard


# ### How to Use t-SNE Effectively
# https://distill.pub/2016/misread-tsne/
# 
# ---
# 
# 

# ## Topic Modeling with Latent Dirichlet Allocation (LDA)
# https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2

# This will be my next excercise in NLP.
