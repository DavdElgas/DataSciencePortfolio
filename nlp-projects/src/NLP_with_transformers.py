#!/usr/bin/env python
# coding: utf-8

# # This is a work in progress

# ## Generative AI

# Creating a body of text to traing a language model.
# Text will come from a classic car forum powered by https://xenforo.com/.
# The intent is to leverage generative AI to create a "virtual mechanic" to help maintain older cars that have a dwindling set of experts available to turn to.

# ###Design Strategy
# 

# The project will use the GPT-2 pre-trained model. This choice is due to the lower cost. The model will be supplemented with very specific data that is was not included in the training of GPT-2. An overview of the project is as follows:
# 
# Here’s a basic outline on how to get started:
# 
# 1. Install Transformers Library
# Ensure you have the transformers library installed, which provides access to pre-trained models like GPT-2.
# 
# 3. Generate Responses
# You can input a prompt to the model and generate responses. Here’s an example:
# 
# 4. Fine-Tuning (Optional)
# If you have domain-specific data (like your car maintenance data), you can fine-tune GPT-2 on this data. This step is more advanced and requires setting up a training loop, handling data batching, and possibly using GPUs for efficient training.
# 
# 5. Deployment
# Once satisfied with the model's responses, you can integrate it into your application. For testing and development purposes, you can run the model on your local machine or use platforms like Google Colab for more computational power.

# ### Create Enviornment

# In[ ]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install requests')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip3 install gensim==3.6.0')
get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')


# In[ ]:


import pandas as pd

import numpy as np

import requests

from bs4 import BeautifulSoup

import re

import nltk
nltk.download("all")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

from transformers import pipeline


# ### Create corpus for a knowledge base

# In[ ]:


# Create list of URLs to seach an parse

url_list =[
            #'https://e9coupe.com/forum/search/165678/'
            #'https://e9coupe.com/forum/search/165679/'
            #'https://e9coupe.com/forum/search/165680/'
            #'https://e9coupe.com/forum/search/165682/'
            #'https://e9coupe.com/forum/search/165703/'
            #'https://e9coupe.com/forum/search/165704/'
            #'https://e9coupe.com/forum/search/165705/'
            'https://e9coupe.com/forum/search/169041/'
            #'https://e9coupe.com/forum/threads/sway-bar-interchange.44955/'
          ]

# Create a df for the urls
df_url = pd.DataFrame(url_list)

column_names = ['url_list']

# Assign the list to the DataFrame's columns
df_url.columns = column_names

# Export the df
df_url.to_csv('e9coupe_df_url', index=False)


# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def fetch_and_parse_thread(thread_url, thread_title, processed_posts):
    numeric_thread_id = re.findall(r'\d+', thread_url.split('/')[-2])[0] if re.findall(r'\d+', thread_url.split('/')[-2]) else 'N/A'

    response = requests.get(thread_url)
    thread_data = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')

        for article in articles:
            post_id = article.get('id', 'N/A')
            numeric_post_id = re.findall(r'\d+', post_id)[0] if re.findall(r'\d+', post_id) else 'N/A'

            if numeric_post_id not in processed_posts:
                processed_posts.add(numeric_post_id)
                content = article.find('div', class_='bbWrapper').get_text(strip=True)
                timestamp = article.find('time', class_='u-dt').get_text(strip=True) if article.find('time', class_='u-dt') else 'N/A'
                post_number_element = article.find('ul', class_='message-attribution-opposite').find('li').find_next_sibling('li')
                post_number = post_number_element.get_text(strip=True) if post_number_element else 'N/A'
                post_number = post_number.lstrip('#') if post_number != 'N/A' else post_number

                thread_data.append({
                    'thread_id': numeric_thread_id,
                    'thread_title': thread_title,  # Use the passed thread title
                    'post_id': numeric_post_id,
                    'timestamp': timestamp,
                    'post_number': post_number,
                    'post_raw': content
                })

    return thread_data

def fetch_thread_urls_from_search(search_url):
    response = requests.get(search_url)
    thread_info = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='contentRow-main')
        for result in results:
            link = result.find('a', href=True)
            title_element = result.find('h3', class_='contentRow-title')
            if link and title_element:
                full_url = 'https://e9coupe.com' + link['href']
                thread_title = title_element.get_text(strip=True)
                thread_info.append((full_url, thread_title))

    return thread_info

def process_all_threads(search_url):
    thread_info = fetch_thread_urls_from_search(search_url)
    processed_posts = set()
    all_data = []

    for url, title in thread_info:
        thread_data = fetch_and_parse_thread(url, title, processed_posts)
        all_data.extend(thread_data)

    return pd.DataFrame(all_data)

# URL of the search results page
search_url = "https://e9coupe.com/forum/search/169041/"

# Process all threads and store in a DataFrame
df = process_all_threads(search_url)

# Display the DataFrame
print(df)

# Export the df
df.to_csv('e9coupe_df_raw.csv', index=False)


# ### Clean the text

# In[ ]:


# Tokenization
post_tokens = []

for text in df['post_raw']:
    post_token = word_tokenize(text.lower())  # Tokenize each post
    post_token = [token for token in post_token if len(token) > 1]  # Filter out single-character tokens
    post_tokens.append(post_token)

df['post_tokens'] = post_tokens  # Add tokenized posts as a new column in the DataFrame

# Stop words
mystopwords = set(stopwords.words("english"))
post_filtered = []

for tokens in df['post_tokens']:
    filtered_tokens = [token for token in tokens if token not in mystopwords and token.isalpha()]
    post_filtered.append(filtered_tokens)

df['post_filtered'] = post_filtered  # Add filtered tokens as a new column

# Lemmatization
lemmatizer = WordNetLemmatizer()

lemm_list = []

for words in df['post_filtered']:
    lemm_words = [lemmatizer.lemmatize(word) for word in words]  # Corrected list comprehension
    lemm_list.append(lemm_words)

df['post_lemm'] = lemm_list  # Add lemmatized words as a new column

# Export the DataFrame to a CSV file
df.to_csv('e9coupe_df.csv', index=False)


# In[ ]:


# Final df

df['thread_id'] = df['thread_id'].fillna(0).astype(int)
df['post_id'] = df['post_id'].fillna(0).astype(int)
df['post_number'] = df['post_number'].str.replace(r'\D', '', regex=True).astype(int)


# In[ ]:


# Take a look

df.info()


# ### Database Schema

# 
# Table 1: Issues
# <br>
# *   Key Should be from pandas
# *   Issue ID (Foreign Key) Should be taken from the thread_key
# *   Issue Should be the thread title
# *   Short Description Should be from the thread post
# *   Keywords: Should be from the thread post
# 
# Table 2: Solutions
# <br>
# *   Key Should be from pandas
# *   Issue ID (Foreign Key) Should be taken from the thread_key
# *   Detailed Solution Should be from the thread posts
# *   Keywords Should be from the thread posts
# 
# Table 3: Notes
# <br>
# *   Key Should be from pandas
# *   Issue ID (Foreign Key) Should be taken from the thread_key
# *   Unstructured Note Content Should be from the thread posts

# ####Table 1: Issues
# 
# *   ID (Unique ID)
# *   Issue
# *   Short Description
# *   Keywords
# 
# 
# 

# In[ ]:


#Table 1 Issues

#*   Key Should be from pandas
#*   Issue ID (Foreign Key) Should be taken from the thread_id
#*   Issue Should be the thread title
#*   Short Description Should be from the thread post
#*   Keywords: Should be from the thread post


df_table_1 = pd.DataFrame()

# Specify the columns you want to copy
columns_to_copy = ['thread_id','thread_title','post_number','post_raw']

# Copy the specified columns
df_table_1[columns_to_copy] = df[columns_to_copy].copy()


# ### Summarization/Short Description
# Very difficult to find a winning strategy here that can accomidate both long and short length text blocks.
# 
# **Extractive Summarization**
# <br>
# Pros:
# <br>
# Good with Raw Text: Extractive methods can work directly with raw, unstructured text, as they mainly focus on selecting key sentences or phrases without needing deep linguistic processing.
# Straightforward Implementation: These methods do not require complex preprocessing like tokenization or lemmatization, simplifying their implementation.
# <br>
# Cons:
# <br>
# Limited Depth in Understanding: While they can handle raw text, they may not fully capture the nuanced meaning, especially when the text contains complex structures or unorthodox language use.
# Less Effective with Poorly Structured Text: In cases where the text is poorly structured or highly informal, extractive summarization might struggle to identify the main points effectively.
# <br>
# <br>
# 
# 
# **Abstractive Summarization** (like sshleifer/distilbart-cnn-12-6)
# <br>
# Pros:
# <br>
# Advanced Processing Capabilities: Abstractive models, especially those based on transformer architectures, are designed to handle and interpret raw text, capturing deeper linguistic and contextual nuances.
# Higher Tolerance for Unstructured Text: These models can manage unstructured or informal text by understanding and then rephrasing it in a more coherent and structured summary.
# <br>
# Cons:
# <br>
# Dependence on Preprocessing for Optimal Performance: While they can process raw text, the quality of the output can be significantly improved with proper tokenization and lemmatization, especially for complex texts.
# Potential Overhead: Requires more computational resources to process and understand raw text, which might be more efficiently handled with some level of preprocessing.
# <br>
# <br>
# **Hybrid Summarization**
# <br>
# Pros:
# <br>
# Flexibility in Text Processing: Combining extractive and abstractive methods allows for handling both raw and preprocessed text, adapting to the text's structure and complexity.
# Balanced Approach: Can leverage the strengths of extractive methods in handling raw text for identifying key points, while using abstractive techniques for generating a coherent summary.
# <br>
# Cons:
# <br>
# Complex Preprocessing Requirements: The need to integrate both extractive and abstractive approaches may necessitate more sophisticated preprocessing strategies to optimize performance.
# Potential for Processing Inefficiencies: The combined approach might lead to redundancies or inefficiencies in processing, especially if the text is either too raw or overly preprocessed.
# <br>
# <br>
# Landed on a transformer model.

# ###Summary

# In[ ]:


from transformers import pipeline

# Initialize the summarizer (make sure to have transformers installed)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    min_threshold = 0  # Threshold for text length
    if len(text.split()) < min_threshold:
        return text  # Return original for short texts
    else:
        try:
            input_length = len(text.split())
            max_length = max(10, int(input_length / 2))  # Example: half of the input length
            min_length = max(5, int(max_length / 2))    # Ensuring min_length is not too small

            return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        except Exception as e:
            print(f"Error during summarization: {e}")
            return text  # Return original text if summarization fails

# Apply summarization conditionally
#df_table_1['summary'] = df_table_1.apply(lambda x: summarize_text(x['post_raw']) if x['post_number'] == 1 else x['post_raw'], axis=1)
df_table_1['summary'] = df_table_1.apply(lambda x: summarize_text(x['post_raw']) if x['post_number'] == 1 else '', axis=1)

# Export the DataFrame to a CSV file
df_table_1.to_csv('df_table_1.csv', index=False)


# In[ ]:


unique_count = df_table_1['summary'].nunique()
print(unique_count)


# ### Keywords

# Raw text vs pre-processed
# Using Pre-Processed Data
# 
# Fine-Tuning BERT or Similar Models
# Pros:
# 
# Effective with Preprocessed and Raw Text: Can handle both preprocessed (for cleaner, more focused analysis) and raw text (to capture nuances and specific jargon).
# Good with Varied Text Lengths: Capable of managing both short and long texts, though long texts might need to be segmented due to token limits.
# Cons:
# 
# Token Limitations for Long Texts: BERT has a token limit (512 tokens), necessitating additional processing for longer texts.
# Resource-Intensive: Requires significant computational resources, especially for processing and fine-tuning on large datasets.
# 2. Using GPT-3 for Response Generation
# Pros:
# 
# Adaptable to Both Text Types: Can generate responses based on both preprocessed and raw text, utilizing the context effectively.
# Handles Long Text Well: GPT-3's architecture allows it to handle longer sequences of text better than many other models, making it suitable for detailed responses.
# Cons:
# 
# Context Dilution in Long Texts: Might lose context in very long text sequences, affecting response relevance.
# Operational Cost and Accessibility: Using GPT-3, particularly at scale, can be expensive and subject to usage constraints.
# 3. Hybrid Approach (Combining Fine-Tuned BERT and GPT-3)
# Pros:
# 
# Versatility with Text Types: Combines BERT's analytical strength (which can handle both preprocessed and raw text) with GPT-3’s generative capabilities.
# Effective with All Text Lengths: Can manage short texts efficiently and long texts through a combination of BERT's analytical precision and GPT-3's generative scope.
# Cons:
# 
# Integration Complexity: Managing and integrating two systems, especially ensuring coherence between them, adds complexity.
# High Resource Requirement: Both models are resource-intensive, which could be a significant factor, particularly for processing large datasets.
# Additional Consideration: Preprocessed vs. Raw Text
# Preprocessed Text: While cleaner and more structured, may lose crucial context or technical terms, which can be pivotal in a specialized domain like automotive discussions.
# Raw Text: Retains all original information, including specific jargon and nuances, but is more challenging to process due to irregularities and noise.

# In[ ]:


# Keyword extraction with BERT
# Looks better
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to extract keywords using BERT
def bert_extract_keywords(text, tokenizer, model, top_n=5):
    # Tokenize and encode the text
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs['input_ids'][0]

    # Get the embeddings from the last hidden layer
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)

    # Compute word importance by summing up the embeddings
    word_importance = torch.sum(embeddings, dim=1)

    # Get the indices of the top n important words
    top_n_indices = word_importance.argsort(descending=True)[:top_n]

    # Filter out indices that are out of range of input_ids
    top_n_indices = [idx for idx in top_n_indices if idx < len(input_ids)]

    # Decode the top n words
    keywords = [tokenizer.decode([input_ids[idx]]) for idx in top_n_indices]


    return keywords

# Apply the function to each row in 'post_content'
#df_table_1['keywords'] = df_table_1['post_raw'].apply(lambda x: bert_extract_keywords(x, tokenizer, model))
df_table_1['keywords'] = df_table_1.apply(lambda x: bert_extract_keywords(x['post_raw'], tokenizer, model) if x['post_number'] == 1 else [], axis=1)


# Display the DataFrame with extracted keywords
print(df_table_1[['post_raw', 'keywords']])


# In[ ]:


# Finalize table 1
#*   Key Should be from pandas
#*   Issue ID (Foreign Key) Should be taken from the thread_id
#*   Issue Should be the thread title
#*   Short Description Should be from the thread post
#*   Keywords: Should be from the thread post

df_table_1.rename(columns={'thread_id': 'issue_id'}, inplace=True)
df_table_1.rename(columns={'thread_title': 'issue'}, inplace=True)
df_table_1.rename(columns={'summary': 'description'}, inplace=True)
df_table_1.drop('post_raw', axis = 1, inplace=True)
df_table_1.drop('post_number', axis = 1, inplace=True)

df_issue = df_table_1.copy()

# Export the DataFrame to a CSV file
df_issue.to_csv('df_issue.csv', index=False)


# In[ ]:


# Take a look

df_issue.info()


# ####Table 2: Solutions
# 
# *   ID (Unique Note ID)
# *   Issue
# *   Issue ID (Foreign Key linking to ID in Issues table)
# *   Solution
# 

# In[ ]:


#Table 2
#*   Key Should be from pandas
#*   Issue Should be the thread title
#*   Issue ID (Foreign Key) Should be taken from the thread_key
#*   Detailed Solution Should be the concatinated post_raw per thread_id
#*   Keywords Should be from be the concatinated post_raw per thread_id

df_table_2 = pd.DataFrame()

# Specify the columns you want to copy
columns_to_copy = ['thread_id','thread_title']#, 'post_raw']

# Copy the specified columns
df_table_2[columns_to_copy] = df[columns_to_copy].copy()

df_table_2 = df_table_2[['thread_id', 'thread_title']].drop_duplicates()


# In[ ]:


# Solutions and Keywords need to use concatenated post_raw values each post per thread
def concatenate_posts_by_thread(df, group_col='thread_id', concat_col='post_raw'):

    # Group by 'group_col' and concatenate 'concat_col' for each group
    concatenated_df = df.groupby(group_col)[concat_col].apply(' '.join).reset_index()

    # Rename the column to make it more descriptive
    concatenated_column_name = f'concatenated_{concat_col}'
    concatenated_df.rename(columns={concat_col: concatenated_column_name}, inplace=True)

    return concatenated_df

# Assuming df is your DataFrame
concatenated_posts_df = concatenate_posts_by_thread(df)

# Display the DataFrame with concatenated posts
print(concatenated_posts_df)


# In[ ]:


concatenated_posts_df.info()


# In[ ]:


# Finalize table 2
#*   Key Should be from pandas
#*   Issue Should be the thread title
#*   Issue ID (Foreign Key) Should be taken from the thread_key
#*   Detailed Solution Should be the concatinated post_raw per thread_id
#*   Keywords Should be from be the concatinated post_raw per thread_id

df_solutions = pd.merge(df_table_2, concatenated_posts_df, on='thread_id')

df_solutions.rename(columns={'thread_title': 'issue'}, inplace=True)
df_solutions.rename(columns={'thread_title': 'issue'}, inplace=True)



# In[ ]:


# Export the DataFrame to a CSV file
df_solutions.to_csv('df_solutions.csv', index=False)


# In[ ]:


# Take a look

df_solutions.info()


# In[ ]:


# Table 3: Notes

#*   Key Should be from pandas
#*   Issue ID (Foreign Key) Should be taken from the thread_key
#*   Unstructured Content: Should be from the rawmthread posts

df_table_3 = pd.DataFrame()

# Specify the columns you want to copy
columns_to_copy = ['thread_id','post_raw']

# Copy the specified columns
df_table_3[columns_to_copy] = df[columns_to_copy].copy()


df_notes = df_table_3.copy()

# Export the DataFrame to a CSV file
df_notes.to_csv('df_notes.csv', index=False)


# In[ ]:


# Take a look

df_notes.info()


# ## Load tables to Snowflake

# What if I processed a complete auto repair guide to train on ??
# 
# ChatGPT
# Training a keyword extraction model on a complete auto repair guide or similar domain-specific text is an excellent idea, especially if your goal is to improve the relevance and accuracy of keyword extraction for automotive discussions. Here's how you can approach this:
# 
# Domain-Specific Corpus: Obtain a corpus of domain-specific texts, such as auto repair manuals, guides, or technical documents. This corpus should contain text that is closely related to the automotive domain and covers various aspects of auto repair, maintenance, and troubleshooting.
# 
# Preprocessing: Preprocess the text data, similar to what you did with your forum data. This includes tokenization, stop word removal, lemmatization, and any other necessary cleaning steps.
# 
# Keyword Labeling: Manually label or annotate the relevant keywords in this domain-specific text. You can do this by identifying words or phrases that are highly relevant to specific topics, parts, or procedures in the auto repair domain. This labeled data will serve as your training dataset.
# 
# Training a Keyword Extraction Model: Utilize a machine learning or NLP model to train on this labeled dataset. You can explore different models, including traditional machine learning models (e.g., SVM, Random Forest) or deep learning models (e.g., BERT with fine-tuning). The model's objective should be to predict or extract keywords based on the labeled examples.
# 
# Evaluation: After training the model, evaluate its performance using appropriate metrics. You can use metrics such as precision, recall, and F1-score to assess how well the model identifies keywords in the automotive domain.
# 
# Fine-Tuning: Depending on the evaluation results, you may need to fine-tune the model to improve its keyword extraction accuracy. This can involve adjusting hyperparameters, increasing the amount of training data, or refining the labeling process.
# 
# Integration: Once you have a well-trained keyword extraction model, you can integrate it into your project. You can use this model to extract keywords from the automotive forum discussions, improving the relevance and specificity of the extracted keywords.
# 
# Continuous Learning: Keep in mind that language and terminology in the automotive field can evolve over time. It's essential to periodically update and retrain your keyword extraction model to stay current with the latest industry terminology and trends.
# 
# By training a model on a domain-specific corpus, you can leverage the specialized knowledge contained in auto repair guides to enhance the accuracy and context-awareness of your keyword extraction process. This approach should help you generate more relevant and meaningful keywords for automotive discussions, ultimately improving the quality of your analysis and insights.

# ## Model creation and training

# In[ ]:


from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


# In[ ]:


from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set tokenizer padding token with 'left' padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Define your prompt for generating a pizza recipe
prompt = "How do I replace the water pump on a 1973 BMW 3.0cs ?"

# Encode the prompt
inputs = tokenizer.encode(prompt, return_tensors='pt', padding='max_length', max_length=100, truncation=True)

# Generate text using the model
outputs = model.generate(inputs, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode and print the generated recipe
generated_recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_recipe)


# ## Parking Lot

# In[ ]:


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to extract keywords using BERT
def bert_extract_keywords(text, tokenizer, model, top_n=5):
    # Tokenize and encode the text
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs['input_ids'][0]

    # Get the embeddings from the last hidden layer
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)

    # Compute word importance by summing up the embeddings
    word_importance = torch.sum(embeddings, dim=1)

    # Get the indices of the top n important words
    top_n_indices = word_importance.argsort(descending=True)[:top_n]

    # Filter out indices that are out of range of input_ids
    top_n_indices = [idx for idx in top_n_indices if idx < len(input_ids)]

    # Decode the top n words
    keywords = [tokenizer.decode([input_ids[idx]]) for idx in top_n_indices]

    return keywords

# Apply the function to each row
concatenated_posts_df['keywords'] = concatenated_posts_df['concatenated_post_raw'].apply(lambda x: bert_extract_keywords(x, tokenizer, model))


# In[ ]:


# Key word extraction with Tfid
# Results are fast but terrible

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=100)

# Fit the vectorizer to the entire dataset in df['post_content']
vectorizer.fit(df['post_raw'])

# Function to extract top N keywords for a given text entry
def extract_keywords(text, vectorizer, top_n=5):
    # Transform the text to tf-idf vector
    tfidf_vector = vectorizer.transform([text])

    # Retrieve the scores of the tf-idf vector
    scores = tfidf_vector.toarray().flatten()

    # Get the indices of the top n scores
    top_n_indices = scores.argsort()[-top_n:][::-1]

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Map indices to feature names (keywords)
    keywords = [feature_names[i] for i in top_n_indices]

    return keywords

# Apply the function to each row in 'post_content'
df_table_1['keywords'] = df_table_1['post_raw'].apply(lambda x: extract_keywords(x, vectorizer))

# Display the DataFrame with extracted keywords
print(df_table_1[['post_raw', 'keywords']])


# In[ ]:


# Keyword extraction with Spacy
# Results are good but not great

import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract keywords using spaCy
def spacy_extract_keywords(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract entities and noun chunks as keywords
    # You can adjust this to suit your needs (e.g., include or exclude certain POS tags or entities)
    keywords = [chunk.text for chunk in doc.noun_chunks] + [ent.text for ent in doc.ents]

    return keywords

# Apply the function to each row in 'post_content'
df['keywords'] = df['post_raw'].apply(spacy_extract_keywords)

# Display the DataFrame with extracted keywords
print(df[['post_raw', 'keywords']])


# In[ ]:


# Summarization/Short Description

# Hugging face token
#XXXXXXXXXXXXXX

# Initialize the summarizer (make sure to have transformers installed)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    min_threshold = 50  # Threshold for text length
    if len(text.split()) < min_threshold:
        return text  # Return original for short texts
    else:
        try:
            # Dynamically adjust max_length based on the input length
            input_length = len(text.split())
            max_length = max(30, int(input_length / 2))  # Example: half of the input length
            min_length = max(10, int(max_length / 2))    # Ensuring min_length is not too small

            return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        except Exception as e:
            print(f"Error during summarization: {e}")
            return text  # Return original text if summarization fails

# Assuming df_info is your DataFrame and 'post_content' is the column with texts
summarized_content = [summarize_text(text) for text in df['post_raw']]

# Add the summarized content as a new column
df_table_1['summary'] = summarized_content

# Export the DataFrame to a CSV file
df_table_1.to_csv('df_table_1.csv', index=False)


# In[ ]:


# Dec 27 Version
# Initialize lists
thread_keys = []
bread_crumbs = []
titles = []
post_keys = []
post_contents = []
post_authors = []

# Assuming df_url['url_list'] is already defined and contains the URLs to scrape
for url in df_url['url_list']:
    page = requests.get(url)  # Send an HTTP GET request to the current URL
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all('div', class_='contentRow-main')

    for result in results:
        title_element = result.find('h3', class_='contentRow-title')
        if title_element and title_element.find('a'):
            url_tail = title_element.find('a')['href']
            detailed_url = 'https://e9coupe.com' + url_tail

            page_detail = requests.get(detailed_url)  # Request for detailed page
            soup_detail = BeautifulSoup(page_detail.content, "html.parser")

            # Extract each breadcrumb item
            breadcrumb_items = []
            breadcrumbs = soup_detail.find('ul', class_='p-breadcrumbs')
            if breadcrumbs:
                for li in breadcrumbs.find_all('li', itemprop='itemListElement'):
                    name = li.find('span', itemprop='name')
                    if name:
                        breadcrumb_items.append(name.get_text())

            # Find thread key
            html_tag = soup_detail.find('html')
            if html_tag and html_tag.has_attr('data-content-key'):
                thread_id = html_tag['data-content-key']
                match = re.search(r'\d+', thread_id)
                thread_key = match.group() if match else 'none'
            else:
                thread_key = 'none'

            # Find title
            title_tag = soup_detail.find('title')
            if title_tag:
                thread_title = title_tag.get_text().split('|')[0].strip()
            else:
                thread_title = 'Title not found'

            # Parse soup into articles for this particular thread
            post_articles = soup_detail.find_all('article', class_='message message--post js-post js-inlineModContainer')

            for article in post_articles:
                post = article['data-content']
                post_id = post.split("-")[1]
                post_author = article['data-author']
                post_content = article.find('div', class_='bbWrapper').get_text(strip=True)

                # Append the current post data
                post_keys.append(post_id)
                post_contents.append(post_content)
                post_authors.append(post_author)

                # Append the current thread_key, title, and breadcrumbs for this post
                thread_keys.append(thread_key)
                titles.append(thread_title)
                bread_crumbs.append(breadcrumb_items)

# Create DataFrame
df = pd.DataFrame({
    'thread_key': thread_keys,
    'title': titles,
    'bread_crumb': bread_crumbs,
    'post_key': post_keys,
    'author': post_authors,
    'post_content': post_contents
})

# Export the DataFrame to a CSV file
#df.to_csv('e9coupe_df.csv', index=False)


# In[ ]:


# Assuming df is already defined and contains a column 'post_content'

# Text tokenization
post_tokens = []

for text in df['post_content']:
    post_token = word_tokenize(text.lower())  # Tokenize each post
    post_token = [token for token in post_token if len(token) > 1]  # Filter out single-character tokens
    post_tokens.append(post_token)

df['post_tokens'] = post_tokens  # Add tokenized posts as a new column in the DataFrame

# Remove stop words
mystopwords = set(stopwords.words("english"))
post_filtered = []

for tokens in df['post_tokens']:
    filtered_tokens = [token for token in tokens if token not in mystopwords and token.isalpha()]
    post_filtered.append(filtered_tokens)

df['post_filtered'] = post_filtered  # Add filtered tokens as a new column

# Lemmatization
lemmatizer = WordNetLemmatizer()

lemm_list = []

for words in df['post_filtered']:
    lemm_words = [lemmatizer.lemmatize(word) for word in words]  # Corrected list comprehension
    lemm_list.append(lemm_words)

df['post_lemm'] = lemm_list  # Add lemmatized words as a new column

# Export the DataFrame to a CSV file
#df.to_csv('e9coupe_df.csv', index=False)


# ## Finding Broken Image links

# ###Test Cases

# In[ ]:


# Post with no images
# https://e9coupe.com/forum/threads/how-to-find-threads-started-by-x.23058/

# Post with one e9 hosted image
# https://e9coupe.com/forum/threads/ex-han-stuck-csl-batmobile-sells-for-171-661.6560/

# Post with 19 e9 hosted image
# https://e9coupe.com/forum/threads/1966-2000cs.44749/

# Post with 38 mixed hosting imgaes
# X hosted images
# Y broken links
# https://e9coupe.com/forum/threads/interior-and-down-town-la-photoshoot-occoupe-3-0csi-update.23139/


# ###Code

# In[ ]:


# Step:1
# Create list of URLs to seach an parse
# Need a way to programmatically create this list
# I think to automate this I would need to access the API.


url_list =[
            #'https://e9coupe.com/forum/search/165678/'
            #'https://e9coupe.com/forum/search/165679/'
            #'https://e9coupe.com/forum/search/165680/'
            #'https://e9coupe.com/forum/search/165682/'
            #'https://e9coupe.com/forum/search/165703/'
            #'https://e9coupe.com/forum/search/165704/'
            #'https://e9coupe.com/forum/search/165705/'
            'https://e9coupe.com/forum/search/168221/'
          ]

# Create a df for the urls
df_1 = pd.DataFrame(url_list)

column_names = ['url_list']#, 'Column_2', 'Column_3', ...]  # Add as many names as there are columns

# Assign the list to the DataFrame's columns
df_1.columns = column_names


# Export the df
# df_1.to_csv('e9coupe_df_1', index=False)

# print(df_1)


# In[ ]:


# This works to parse a post


# Set the base URL
URL_base = 'https://e9coupe.com/forum/threads/jeremy-clarkson-reveals-the-car-he-regrets-selling-the-most.37009/?page=1'

# Initialize an empty list to store data from all pages
all_data = []

# Create a DataFrame from the collected dataa
df = pd.DataFrame()

# Send an HTTP GET request to the base URL
page = requests.get(URL_base)

# Parse the HTML content
soup_base = BeautifulSoup(page.content, "html.parser")

# Find the script tag containing the JSON data
script_tag = soup_base.find('script', type='application/ld+json')

# Extract the JSON data from the script tag (assuming it always exists)
json_data = json.loads(script_tag.string)

# Get the number of posts from the JSON data
posts = json_data['interactionStatistic']['userInteractionCount']

# Calculate the number of pages (assuming 20 posts per page)
pages = math.ceil(posts / 20)

# Find OP and URL
script_tag = soup_base.find('script', type='application/ld+json')
json_data = json.loads(script_tag.string)

headline = json_data['headline']
published = json_data['datePublished']
date_part = published.split("T")[0]
author = json_data['author']['name']
link_to_post = json_data['mainEntityOfPage']['@id']
posts = json_data['interactionStatistic']['userInteractionCount'] + 1


# Export the DataFrame to a CSV file
# df.to_csv('e9coupe_test.csv', index=False)

# Loop to increment and parse the URLs
for i in range(1, pages + 1):
    URL = URL_base + 'page-' + str(i)
    # Send an HTTP GET request to the current URL
    page = requests.get(URL)

    # Parse the HTML content of the current page
    soup = BeautifulSoup(page.content, "html.parser")

    # Find all post articles with the specified class attribute
    post_articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')

    # Iterate through the posts on the current page
    for article in post_articles:
        post_id = article['data-content']
        post_author = article['data-author']

        # Find attachments within the current post
        att_tags = article.find_all('a', class_='file-preview js-lbImage')
        total_att_count = len(att_tags)

        # Find all images within the current post and count them
        img_tags = article.find_all('img', class_='bbImage')
        total_image_count = sum(1 for tag in img_tags)

        # Filter e9 hosted images and count them
        e9_hosted_image_count = sum(1 for tag in img_tags if 'e9coupe.com' in tag.get('src'))

        # Filter 3rd party hosted images and count them
        third_party_image_count = sum(1 for tag in img_tags if 'forum/proxy.php' in tag.get('src'))

        # Append the extracted values to the all_data list
        all_data.append({
            'post_id': post_id,
            'post_author': post_author,
            'all_images': total_image_count,
            'hosted_images': e9_hosted_image_count,
            'linked_images': third_party_image_count,
            'attachments': total_att_count,
        })

# Create a DataFrame from the collected data
df = pd.DataFrame(all_data)

# Count the number of posts processed
num_articles = len(df)

# Print report header
print(f"Headline:        {headline}")
print(f"Author:          {author}")
print(f"URL:             {link_to_post}")
print(f"Published:       {date_part}")

print(f"Total posts:     {posts}")
print(f"Processed posts: {num_articles}")
print()

# Print the DataFrame
print(df)

# Export the DataFrame to a CSV file
# df.to_csv('e9coupe_test.csv', index=False)


# In[ ]:


# Create a DataFrame from the collected dataa
df = pd.DataFrame()

# Initialize lists
breadcrumb_items = []
thread_keys = []
titles = []
post_keys = []
post_contents = []
post_authors = []

# Parse the search page into URLs
for url in df_url['url_list']:
  page = requests.get(url) # Send an HTTP GET request to the current URL

  soup = BeautifulSoup(page.content, "html.parser")
  results = soup.find_all('div', class_='contentRow-main')

  for link in results:
      title_element = link.find('h3', class_='contentRow-title')
      if title_element and title_element.find('a'):
          title = title_element.get_text(strip=True)
          url_tail = title_element.find('a')['href']
          url = 'https://e9coupe.com'+url_tail

          page = requests.get(url) # Send an HTTP GET request to the current URL
          soup = BeautifulSoup(page.content, "html.parser") # Parse the HTML content of the current page


          # Find the breadcrumb list
          breadcrumbs = soup.find('ul', class_='p-breadcrumbs')

          # Extract each breadcrumb item
          if breadcrumbs:
              for li in breadcrumbs.find_all('li', itemprop='itemListElement'):
                  name = li.find('span', itemprop='name')
                  if name:
                      breadcrumb_items.append(name.get_text())
              bread_crumbs.append(breadcrumb_items) # Append to a list


          # Find title
          title_tag = soup.find('title')
          if title_tag:
              thread_title = title_tag.get_text().split('|')[0].strip()
          else:
              thread_title = 'Title not found'

          titles.append(thread_title) # Append to a list



          # Find  thread key
          html_tag = soup.find('html') # Find the <html> tag

          # Extract the value of the 'data-content-key' attribute
          if html_tag and html_tag.has_attr('data-content-key'):
              thread_id = html_tag['data-content-key']
              # Use a regular expression to extract the number value
              match = re.search(r'\d+', thread_id)
              if match:
                thread_key = match.group()

          else:
              thread_key = 'none'


          thread_keys.append(thread_key) # Append to a list


          # Parse soup into articles
          post_articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')

          # Iterate through the articles to get the post information
          for article in post_articles:
              post_id = article['data-content']
              post_author = article['data-author']
              post_content = article.find('div', class_ = "bbWrapper").get_text(strip=True)

              # Append to a list
              post_keys.append(post_id)
              post_contents.append(post_content)
              post_authors.append(post_author)


df = pd.DataFrame({
    'bread_crumb': [bread_crumbs[0]] * len(post_keys),
    'thread_keys': [thread_keys[0]] * len(post_keys),
    'title': [titles[0]] * len(post_keys),  # Repeat the title for each post
    'post_key': post_keys,
    'author': post_authors,
    'post_content': post_contents
})


# Export the DataFrame to a CSV file
# df.to_csv('e9coupe_df.csv', index=False)


# In[ ]:





# In[ ]:


# Step 2: Parse soup into articles
# There is a thread and post key in each article
# <a href="/forum/threads/wtb-vgs-slanted-intake-manifold.37038/post-314616" rel="nofollow">

pages = 3

# Initialize an empty list to store URLs of all posts
post_urls = []

# Loop to increment and parse the URLs

for value in df_url['url_list']:
  for i in range(1, pages + 1):
      URL = URL_base + 'page-' + str(i)
      # Send an HTTP GET request to the current URL
      page = requests.get(URL)

      # Parse the HTML content of the current page
      soup = BeautifulSoup(page.content, "html.parser")

      # Find all post articles in the source
      post_articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')

      # Iterate through the posts on the current page
      for article in post_articles:
          print(url)
          post_id = article['data-content']
          print(post_id)
          post_author = article['data-author']
          print(post_author)
          post_content = article.find('div', class_ = "bbWrapper").get_text(strip=True)
          print(post_content)
          print()
          #print(article)
          print('____________________________________________________________')


# In[ ]:


# testing various parsing logic
import requests
from bs4 import BeautifulSoup

# Web URL
Web_url = "https://e9coupe.com/forum/threads/holiday-time-has-come.44824/"

# Get URL Content
r = requests.get(Web_url)

# Parse HTML Code
soup = BeautifulSoup(r.content, 'html.parser')
print(soup.prettify())


# Here is the header where you can get the thread ID. Should be only once per page
<html id="XF" lang="en-US" dir="LTR"
	data-app="public"
	data-template="thread_view"
	data-container-key="node-10"
	data-content-key="thread-44782"
	data-logged-in="true"
	data-cookie-prefix="xf_"
	data-csrf="1705990108,264f4a124e9a97ea9c31e95179e29795"
	class="has-no-js template-thread_view"
	>



# Thread level Metadata
#headline": "Holiday Time Has Come.",
#    "articleBody": "Hi Members.  It's the time of year in this stressful life we all seem to be in to take time to dwell on all the nice times we have had in our lifetime. Take time to love your family, your friends and your neighbors. it's time to realize how lucky...",
#    "articleSection": "Off-Topic",
#    "author": {
#        "@type": "Person",
#        "name": "Koopman"

# See ifthis is once per "page" to get the threadID
#<option value='{"search_type":"post","c":{"thread":44824}}'>







# After extracting the threadID, get the articles for that threadID
# I think this is the "article" delimiter
#<div class="message-content js-messageContent">#
# You can get a post ID here:
#<div class="message-content js-messageContent">
#<div class="message-userContent lbContainer js-lbContainer" data-lb-caption-desc="Koopman · Dec 24, 2023 at 6:45 PM" data-lb-id="post-394598">


# This would allow me to join the post to the thread, and then use the thread name for classification, I think


# In[ ]:


# Correctly classifies posts
# Just need to find the post_id to join it back to step1


import re

pages = 1


for value in df_url['url_list']:
    for i in range(1, pages + 1):
        url = value + '?page=' + str(i)
        page = requests.get(url)

        # Parse the HTML content of the page
        soup = BeautifulSoup(page.content, "html.parser")

for result in soup.find_all('div', class_='contentRow-main'):
    title_element = result.find('h3', class_='contentRow-title')
    if title_element and title_element.find('a'):
        title = title_element.get_text(strip=True)
        url = title_element.find('a')['href']

        # Extract the integer following the period (.) using regex
        thread_match = re.search(r'\.(\d+)', url)
        thread_id = thread_match.group(1) if thread_match else None

        # Extract the integer following "post-" using regex
        post_match = re.search(r'post-(\d+)', url)
        post_id = post_match.group(1) if post_match else None



        # Check if it's a thread or a post
        post_info = result.find('div', class_='contentRow-minor').get_text(strip=True)
        if "Thread" in post_info:
            type = "Thread"
        elif "Post" in post_info:
            type = "Post"
        else:
            type = "Unknown"

        print(f"Title: {title}, URL: {url}, Type: {type}")
        print(f"ThreadID: {thread_id}, PostID: {post_id}")

        print('--------------------------------------------------')

        # Append the information to a list
        post_urls.append({'title': title, 'url': url, 'type': type})


# In[ ]:


# Step 1: Parse soup into articles

# Set the base URL
URL_base = 'https://e9coupe.com/forum/threads/wtb-vgs-slanted-intake-manifold.37038/?page=1'

# Initialize an empty list to store data from all pages
all_data = []

# Create a DataFrame from the collected dataa
df = pd.DataFrame()

# Parse the HTML content
soup_base = BeautifulSoup(page.content, "html.parser")


# Export the DataFrame to a CSV file
df.to_csv('e9coupe_test.csv', index=False)

# Set the maximum number of pages to search in each URL
pages = 3

# Initialize an empty list to store URLs of all posts
post_urls = []

# Loop to increment and parse the URLs

for url in df_url['url_list']:
  for i in range(1, pages + 1):
      URL_full = url + 'page-' + str(i)
      # Send an HTTP GET request to the current URL
      page = requests.get(URL_full)


      # Parse the HTML content of the current page
      soup = BeautifulSoup(page.content, "html.parser")

      # Find all post articles with the specified class attribute
      post_articles = soup.find_all(class_='message message--post js-post js-inlineModContainer')

      # Iterate through the posts on the current page
      for article in post_articles:
          post_id = article['data-content']
          post_author = article['data-author']
          #post_content = soup.find('div', class_ = "bbWrapper")
          post_content = article.find('div', class_ = "bbWrapper").get_text(strip=True)
          print(post_id,post_author)
          print(post_content)
          print('____________________________________________________________')


# In[ ]:


# Step:2 Find and classify posts as threads

# Set the maximum number of pages to search in each URL
pages = 1

# Initialize an empty list to store URLs of all posts
post_urls = []

for value in df_url['url_list']:
  for i in range(1, pages + 1):
      URL = URL_base + 'page-' + str(i)
      # Send an HTTP GET request to the current URL
      page = requests.get(URL)

      # Parse the HTML content of the current page
      soup = BeautifulSoup(page.content, "html.parser")

      # Parse Soup to find articles
      post_articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')

      # Parse Soup to find post class information
      post_class = soup.find_all('div', class_='contentRow-main')

      # Iterate through the article and extract the postID, Author and content
      for article in post_articles:
          post_id = article['data-content']
          post_author = article['data-author']
          print(post_id)
          print(post_author)
          #post_content = article.find('div', class_ = "bbWrapper").get_text(strip=True)
          #print(post_content)
          print('_________________End_of_first_for_loop_____________')

      print('Start of second for loop')
      print(f"Number of elements in post_class: {len(post_class)}")
      # Iterate through the article and classify the content
      for post in post_class:
      # Check if it's a thread or a post
          post_info = post.find('div', class_='contentRow-minor').get_text(strip=True)
          if "Thread" in post_info:
              type = "Thread"
          elif "Post" in post_info:
              type = "Post"
          else:
              type = "Unknown"
          print(type)



# Create a DataFrame for the URLs
#df_2 = pd.DataFrame(post_urls)

# Export the DataFrame to a CSV file
#df_2.to_csv('e9coupe_df_2.csv', index=False)


# In[ ]:


# Step 1. Initial parsing of the source
pages = 3

# Initialize an empty list to store URLs of all posts
post_urls = []

# Loop to increment and parse the URLs

for value in df_url['url_list']:
  for i in range(1, pages + 1):
      URL = URL_base + 'page-' + str(i)
      # Send an HTTP GET request to the current URL
      page = requests.get(URL)

      # Parse the HTML content of the current page
      soup = BeautifulSoup(page.content, "html.parser")

      # Find all post articles in the source
      post_articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')

      # Iterate through the posts on the current page
      for article in post_articles:
          print(article)
          #post_id = article['data-content']
          #post_author = article['data-author']
          #print(post_id)
          #print(post_author)
          #post_content = article.find('div', class_ = "bbWrapper").get_text(strip=True)
          #print(post_content)
          print('____________________________________________________________')



# In[ ]:


import requests
from bs4 import BeautifulSoup

# Assuming df_url['url_list'] and URL_base are defined correctly
# Set the maximum number of pages to search in each URL
pages = 3

# Loop to increment and parse the URLs
for value in df_url['url_list']:
    for i in range(1, pages + 1):
        URL = URL_base + 'page-' + str(i)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")

        print(f"Processing URL: {URL}")  # Debugging line

        # Find all post articles in the source
        #articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')
        #print(f"Found {len(articles)} articles")  # Debugging line

        #for article in articles:
        #    post_id = article.get('data-content', 'N/A')
        #    post_author = article.get('data-author', 'N/A')
        #    print(f"Post ID: {post_id}, Author: {post_author}")

        # Classify the type of post
        posts = soup.find_all('div', class_='contentRow-main')
        print(f"Found {len(posts)} posts to classify")  # Debugging line

        for post in posts:
            title_element = post.find('h3', class_='contentRow-title')
            if title_element and title_element.find('a'):
                title = title_element.get_text(strip=True)
                url = title_element.find('a')['href']
                post_info = post.find('div', class_='contentRow-minor').get_text(strip=True)
                if "Thread" in post_info:
                    type = "Thread"
                elif "Post" in post_info:
                    type = "Post"
                else:
                    type = "Unknown"
                print(f"Title: {title}, URL: {url}, Type: {type}")

        print('End of page processing')
        print('____________________________________________________________')


# In[ ]:


results = soup.find_all('div', class_='contentRow-main')

for result in results:
    title_element = result.find('h3', class_='contentRow-title')
    if title_element and title_element.find('a'):
        title = title_element.get_text(strip=True)
        url = title_element.find('a')['href']

        # Check if it's a thread or a post
        post_info = result.find('div', class_='contentRow-minor').get_text(strip=True)
        if "Thread" in post_info:
            type = "Thread"
        elif "Post" in post_info:
            type = "Post"
        else:
            type = "Unknown"

        print(f"Title: {title}, URL: {url}, Type: {type}")
        print('--------------------------------------------------')

        # Append the information to a list
        post_urls.append({'title': title, 'url': url, 'type': type})


# In[ ]:


# Step:1
# Create list of URLs to seach an parse
# Need a way to programmatically create this list
# I think to automate this I would need to access the API.


url_list =[
            #'https://e9coupe.com/forum/search/165678/'
            #'https://e9coupe.com/forum/search/165679/'
            #'https://e9coupe.com/forum/search/165680/'
            #'https://e9coupe.com/forum/search/165682/'
            #'https://e9coupe.com/forum/search/165703/'
            #'https://e9coupe.com/forum/search/165704/'
            #'https://e9coupe.com/forum/search/165705/'
            'https://e9coupe.com/forum/search/168221/'
          ]

# Create a df for the urls
df_1 = pd.DataFrame(url_list)

column_names = ['url_list']#, 'Column_2', 'Column_3', ...]  # Add as many names as there are columns

# Assign the list to the DataFrame's columns
df_1.columns = column_names


# Export the df
df_1.to_csv('e9coupe_df_1', index=False)

print(df_1)



# Step:2 Find threads

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set the maximum number of pages to search in each URL
pages = 3

# Initialize an empty list to store URLs of all posts
thread_urls = []

for value in df_1['url_list']:
    for i in range(1, pages + 1):
        url = value + '?page=' + str(i)
        page = requests.get(url)

        # Parse the HTML content of the page
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find_all('div', class_='contentRow-main')

        for result in results:
            title_element = result.find('h3', class_='contentRow-title')
            if title_element and title_element.find('a'):
                title = title_element.get_text(strip=True)
                url = title_element.find('a')['href']

                # Check if it's a thread or a post
                post_info = result.find('div', class_='contentRow-minor').get_text(strip=True)
                if "Thread" in post_info:
                    type = "Thread"

                    print(f"Title: {title}, URL: {url}, Type: {type}")
                    print('--------------------------------------------------')
                # Append the information to a list
                    url = 'https://e9coupe.com'+url
                    thread_urls.append({'url': url})
                    #thread_urls.append({'title': title, 'url': url, 'type': type})


# Create a DataFrame for the URLs
df_threads = pd.DataFrame(thread_urls)

# Export the DataFrame to a CSV file
# df_threads.to_csv('e9coupe_df_threads.csv', index=False)


# In[ ]:


# Step:3 Find posts


# Set the base URL
URL_base = 'https://e9coupe.com/forum/threads/wtb-vgs-slanted-intake-manifold.37038/?page=1'

# Initialize an empty list to store data from all pages
all_data = []

# Create a DataFrame from the collected dataa
df = pd.DataFrame()

# Send an HTTP GET request to the base URL
page = requests.get(URL_base)

# Parse the HTML content
soup_base = BeautifulSoup(page.content, "html.parser")

# Find the script tag containing the JSON data
script_tag = soup_base.find('script', type='application/ld+json')

# Extract the JSON data from the script tag (assuming it always exists)
json_data = json.loads(script_tag.string)

# Get the number of posts from the JSON data
posts = json_data['interactionStatistic']['userInteractionCount']

# Calculate the number of pages (assuming 20 posts per page)
pages = math.ceil(posts / 20)

# Find OP and URL
script_tag = soup_base.find('script', type='application/ld+json')
json_data = json.loads(script_tag.string)

headline = json_data['headline']
published = json_data['datePublished']
date_part = published.split("T")[0]
author = json_data['author']['name']
link_to_post = json_data['mainEntityOfPage']['@id']
posts = json_data['interactionStatistic']['userInteractionCount'] + 1


# Export the DataFrame to a CSV file
df.to_csv('e9coupe_test.csv', index=False)

# Set the maximum number of pages to search in each URL
pages = 3

# Initialize an empty list to store URLs of all posts
post_urls = []

# Loop to increment and parse the URLs

for value in df_1['url_list']:
  for i in range(1, pages + 1):
      URL = URL_base + 'page-' + str(i)
      # Send an HTTP GET request to the current URL
      page = requests.get(URL)

      # Parse the HTML content of the current page
      soup = BeautifulSoup(page.content, "html.parser")

      # Find all post articles with the specified class attribute
      post_articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')


      # Iterate through the posts on the current page
      for article in post_articles:
          post_id = article['data-content']
          post_author = article['data-author']
          #post_content = soup.find('div', class_ = "bbWrapper")
          post_content = article.find('div', class_ = "bbWrapper").get_text(strip=True)
          print(post_id,post_author)
          print(post_content)
          print('____________________________________________________________')


# In[ ]:


# Parse the body of the page into articles

url = 'https://e9coupe.com/forum/threads/holiday-time-has-come.44824/'
page = requests.get(url)

# Parse the HTML content of the current page
soup = BeautifulSoup(page.content, "html.parser")

# Find all post articles in the source
post_articles = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')

# Iterate through the posts on the current page
for article in post_articles:
    print(url)
    post_id = article['data-content']
    print(post_id)
    post_author = article['data-author']
    print(post_author)
    post_content = article.find('div', class_ = "bbWrapper").get_text(strip=True)
    print(post_content)
    print()
    #print(article)
    print('____________________________________________________________')


# In[ ]:


# Step: 3
# Find articles from each URL
# Find images from each article
# This might be easier if I had API access, but I guess HTML scraping is all about string searches

div_found_flag = 0
data_url_flag = 0
data_src_flag = 0
image_flag = 0
article_post_meta =[]
df_3 = pd.DataFrame(article_post_meta)

for value in df_2['post_url']:
  url = 'https://e9coupe.com' + value
  page = requests.get(url)

  # Create a soup value form the URL
  soup = BeautifulSoup(page.content, "html.parser")

  # Parse the soup content of the ULR into delimited articles
  article_list = soup.find_all('article', class_='message message--post js-post js-inlineModContainer')#, attrs={'data-author'})

  # Parse the article content of the soup into delimited image lines
  for article in article_list:
    div_found_flag = 0
    data_src_flag = 0
    data_src_image_string_flag = 0
    image_url_raw = ''

    img_divs = article.find_all('div', class_="bbImageWrapper js-lbImage")
    for div in img_divs:
      if div:
        div_found_flag = 1

        if div.get('data-src'):
          data_src_flag = 1

        if 'image' in div.get('data-src'):
              data_src_image_string_flag = 1

    # Build the image URL
      if data_src_image_string_flag == 1:
        image_url_raw = div.get('data-src')
        if 'e9coupe.com/forum/attachments/' not in image_url_raw:
          image_url_encoded = image_url_raw.split('image=')[1].split('&hash=')[0]
          image_url = urllib.parse.unquote(image_url_encoded)
      else:
        image_url = div.get('data-src')

# Ideally the author value would be passed in
# Some of these links are my replies, rather than post. This means I cannot address broken links--the OP would need to.
# Im grabbing a nested div by mistake.

    if div_found_flag == 1 and article['data-author'] == 'David' and 'attachments' not in image_url:
      print('----------------------------------------')
      print('----------------------------------------')
      print('Topic url:  '               ,url)  # Print the URL for each article
      #print('New article')
      post_id = article['data-content']
      print('Post ID.:'                   ,post_id)
      author = article['data-author']
      print('Post author:'                ,author)
      #print('div_found_flag:            ',div_found_flag)
      #print('data_src_flag:             ',data_src_flag)
      #print('data_src_image_string_flag:',data_src_image_string_flag)
      print(image_url)
      # Append the extracted values to the df
      article_post_meta.append({
          'topic_url': url,
          'image_url' : image_url,
          'post_id' : article['data-content'],
          'author' : article['data-author']
      })

# Create a DataFrame from the collected data
df_3 = pd.DataFrame(article_post_meta)

# Export the df
# df_3.to_csv('e9coupe_df_3.csv', index=False)


# In[ ]:


# Step: 4
# Create a report
# I havent updated this with the changes I made to Steps 1-3 so it might not work.

# Count the number of posts processed
num_articles = len(df)


# Print report header
print(f"Headline:        {headline}")
print(f"Author:          {author}")
print(f"URL:             {link_to_post}")
print(f"Published:       {date_part}")

print(f"Total posts:     {posts}")
print(f"Processed posts: {num_articles}")
print()

# Print the DataFrame
print(df)

# Export the DataFrame to a CSV file
# df_4.to_csv('e9coupe_df_4.csv', index=False)

