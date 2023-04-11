import gzip
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

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
          "romance", ]


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
        "genre": genre,
        "book_id": data["book_id"],
        "title": data["title_without_series"],
        "ratings": data["ratings_count"],
    }


books_titles = []
with gzip.open("./data/goodreads_books.json.gz") as f:
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
titles = pd.DataFrame.from_dict(books_titles)
titles["ratings"] = pd.to_numeric(titles["ratings"])
titles["mod_title"] = titles["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
titles["mod_title"] = titles["mod_title"].str.lower()
titles["mod_title"] = titles["mod_title"].str.replace("\s+", " ", regex=True)
titles = titles[titles["mod_title"].str.len() > 0]
titles.to_json("./data/books_titles.json")
print(titles)
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(titles["mod_title"])


def search(query, vectorizer):
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = titles.iloc[indices]
    results = results.sort_values("ratings", ascending=False)

    return results.head(5)


res = search("harry potter and the prisoner of azkaban", vectorizer)
print(res)
