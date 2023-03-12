# NLP Imports
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")  # To fix error on streamlit cloud

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# UI imports
import streamlit as st

# NLP STUFF
# Load headlines data from file
with open("headlines.txt", "r") as f:
    headlines = f.readlines()

# Preprocessing
stopwords = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "of",
    "with",
}
punctuations = r"[^\w\s]"

preprocessed_headlines = []
for headline in headlines:
    headline = re.sub(punctuations, "", headline.lower())
    words = [word for word in headline.split() if word not in stopwords]
    preprocessed_headlines.append(" ".join(words))

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(preprocessed_headlines)


def get_top_headlines_from_query(query_text):
    test_data = word_tokenize(query_text.lower())
    v1 = model.infer_vector(test_data)
    most_similar_idx = model.dv.most_similar(v1, topn=5)
    top_indices = [int(_[0]) for _ in most_similar_idx]
    top_headlines = [headlines[idx] for idx in top_indices]
    return top_headlines


def get_top_headlines_from_query_bow(query_arg):
    query = re.sub(punctuations, "", query_arg.lower())
    query_words = [word for word in query.split() if word not in stopwords]
    query_text = " ".join(query_words)

    # Calculate cosine similarity between query and headlines
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, matrix)

    # Sort similarities and print top-k headlines
    k = 5
    similarities = similarities[0]
    indices = similarities.argsort()[::-1][:k]
    top_headlines = [headlines[i] for i in indices]
    return top_headlines


# Load Trained model
model = Doc2Vec.load("headlines1.d2v.model")
# ***--------***

# UI Stuff
st.title("Text Analytics Assignment - Rafay")

query = st.text_input(
    "Write Query and Press Enter", "Karachi witnessed decrease in terrorism"
)

st.write("The current query is:", query)
st.write("**TOP 5 Matches**")

st.write("**Customized Word2Vec**")
top_headlines = get_top_headlines_from_query(query)
for headline in top_headlines:
    st.write(headline)

st.write("**Customized BoW**")
top_headlines = get_top_headlines_from_query_bow(query)
for headline in top_headlines:
    st.write(headline)
