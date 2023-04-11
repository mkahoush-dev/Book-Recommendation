#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import numpy as np
import re
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA


# In[ ]:


genres = ["fantasy", 
          "fiction", 
          "suspense", 
          "supernatural", 
          "mystery", 
          "conspiracy",
          "comedy",
          "action-adventure",
          "thriller", 
          "crime", 
          "poetry",
          "biography",
          "nonfiction",
          "romance",]

def parse_fields(line):
    data = json.loads(line)
    genre = ""
    for shelve in data["popular_shelves"]: 
        if shelve["name"] in genres: 
            genre = shelve["name"] 
            break
        else: 
            pass    
    return {
        "book_id": data["book_id"], 
        "title": data["title_without_series"], 
        "ratings": data["ratings_count"],
        "average_rating": data["average_rating"],
        "genre": genre
    }


# In[ ]:


books_titles = []
with gzip.open("goodreads_books.json.gz") as f:
    while True:
        line = f.readline()
        if not line:
            break
        fields = parse_fields(line)
        if fields["genre"] == "": 
            continue
        try:
            ratings = int(fields["ratings"])
        except ValueError:
            continue
        if ratings > 5:
            books_titles.append(fields)


# In[ ]:


titles = pd.DataFrame.from_dict(books_titles)
titles["ratings"] = pd.to_numeric(titles["ratings"])
titles["mod_title"] = titles["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
titles["mod_title"] = titles["mod_title"].str.lower()
titles["mod_title"] = titles["mod_title"].str.replace("\s+", " ", regex=True)
titles = titles[titles["mod_title"].str.len() > 0]


# In[ ]:


titles["average_rating"] = titles["average_rating"].astype(float)
print(titles.shape)
titles.head()


# In[ ]:


titles.to_csv("book_titles.csv", index=False)


# In[2]:


titles = pd.read_csv("book_titles.csv")
titles.dropna(inplace=True)


# In[ ]:


with gzip.open("goodreads_reviews_dedup.json.gz") as f:
    data = f.read()


# In[ ]:


data = str(data).replace('\n', '')
data = data.replace('}{', '},{')
data = "[" + data + "]"
reviews = pd.read_json(data)
reviews.head()


# In[3]:


books_reviews = []
book_id_list = list(titles['book_id'].astype(str))
def parse_review_fields(line):
    data = json.loads(line)  
    return {
        "book_id": data["book_id"], 
        "n_votes": data["n_votes"], 
        "review_text": data["review_text"],
    }

with gzip.open("goodreads_reviews_dedup.json.gz") as f:
    while True:
        line = f.readline()
        if not line:
            break
        fields = parse_review_fields(line)
        if (fields['review_text']!='') :
            books_reviews.append(fields)


# In[5]:


# books_reviews = [d for d in books_reviews if d['book_id'] in book_id_list] 
reviews = pd.DataFrame.from_dict(books_reviews)
reviews.head()


# In[6]:


reviews = reviews[reviews['book_id'].isin(book_id_list)]


# In[23]:


reviews.sort_values(by=['n_votes'], ascending=False, inplace=True)


# In[24]:


reviews.drop_duplicates(subset=['book_id'], keep='first', inplace=True)


# In[27]:


# reviews = pd.DataFrame.from_dict(books_reviews)
# reviews["n_votes"] = pd.to_numeric(reviews["n_votes"])
reviews["review_text"] = reviews["review_text"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
reviews["review_text"] = reviews["review_text"].str.lower()
reviews["review_text"] = reviews["review_text"].str.replace("\s+", " ", regex=True)
reviews = reviews[reviews["review_text"].str.len() > 0]
reviews.head()


# In[25]:


reviews.shape


# In[26]:


reviews.to_csv("book_reviews.csv", index=False)


# In[29]:


titles['book_id'] = titles['book_id'].astype(str)
book_title_reviews = pd.merge(titles, reviews, on='book_id', how='inner')
print(book_title_reviews.shape)
book_title_reviews.head()


# In[31]:


book_title_reviews.to_csv("book_title_reviews.csv", index=False)


# In[34]:


book_title_reviews['review_text'][100]
