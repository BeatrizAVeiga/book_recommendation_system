import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch



sbert_model = SentenceTransformer('all-mpnet-base-v2')

def load_data():
    df_books = pd.read_csv(r"C:\Users\biave\Documents\IronHack\Quests\Machine Learning\CleanData\books_w_topics.csv")  # Ensure this file is available
    return df_books

df_books = load_data()

with open(r"C:\Users\biave\Documents\IronHack\Quests\Machine Learning\Models\bertopic_model.pkl", 'rb') as f:
    topic_model = pickle.load(f)

with open(r"C:\Users\biave\Documents\IronHack\Quests\Machine Learning\Models\umap_model.pkl", 'rb') as f:
    umap_model = pickle.load(f)

with open(r"C:\Users\biave\Documents\IronHack\Quests\Machine Learning\Models\hdbscan_model.pkl", 'rb') as f:
    hdbscan_model = pickle.load(f)

embeddings = np.load(r"C:\Users\biave\Documents\IronHack\Quests\Machine Learning\Models\sbert_embeddings.pkl", allow_pickle=True)


def get_sbert_embedding(text):
    """Generates BERT Embeddings"""
    embedding = sbert_model.encode(text)
    return embedding


def get_book_covers(isbn):
    base_url = "https://covers.openlibrary.org/b/isbn/{}.jpg"
    if isbn:
        cover_url = base_url.format(isbn)
        return cover_url
    return "https://via.placeholder.com/150?text=No+Cover"

# Function to recommend books using KMeans + PCA
def recommend_books(user_input, df_books, embeddings, topic_model, preferred_format=None, top_n=10, min_rating=2.0):
    # Get the user's input embedding
    user_input_embedding = get_sbert_embedding(user_input).reshape(1, -1)

    # Calculate similarities between user input and book descriptions
    similarities = cosine_similarity(user_input_embedding, embeddings).flatten()

    # Find the most similar topics to the user's input (using cosine similarity)
    user_topic = topic_model.transform([user_input])[0][0]

    # Filter the books that belong to the same topic
    recommendations = df_books[df_books['topic'] == user_topic]

    # Get the indices of the recommendations
    recommendation_indices = recommendations.index

    # Filter similarities to only include recommendations
    similarities_filtered = similarities[recommendation_indices]

    # Add the similarity score to the recommendations
    recommendations['similarity'] = similarities_filtered

    # Filter by rating
    recommendations = recommendations[recommendations['avg_rating_books'] >= min_rating]

    if format_option:
    
        recommendations = recommendations[recommendations['format'] == format_option]
        
    # Sort by similarity and rating
    recommendations = recommendations.sort_values(by=['similarity', 'avg_rating_books'], ascending=[False, False])

    return recommendations[['title', 'avg_rating_books', 'name', 'description', 'similarity', 'topic', 'isbn', 'format']].head(top_n)

# Title And Logo
st.image("icons/FAVICON.png", width=100)
st.title("WELCOME TO BOOKI!")


# Search bar
user_input = st.text_input("🪄 Describe a book you love, and we'll find you a great match!")

# Dropdown for preferred format
format_option = st.selectbox("📖 Preferred Format:", ["Any", "Paperback", "Hardcover", "Ebook"], index=0)
#stars_option = st.selectbox("⭐ Stars:", ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"])
min_rating = st.slider("⭐ Minimum Rating: ", 0, 5, 4, step=1)

# Search button
if st.button("Find Recommendations"):
    if user_input:
        format_option = None if format_option == "Any" else format_option
        recommendations = recommend_books(user_input, df_books, embeddings, topic_model, top_n=5, min_rating=min_rating)

        if recommendations is not None:
            st.subheader("📖 Recommended Books:")
            for _, row in recommendations.iterrows():
                st.markdown(f"{row['title']}")
                st.image(get_book_covers(row['isbn']), width=150, caption=row['title'])
                st.markdown(f"⭐ {row['avg_rating_books']}")
                st.markdown(f"🖊️ {row['name']}")
                st.markdown(f"📖 {row['format']}")
                st.markdown(f"📊 Similarity: {round(row['similarity'], 2)}")
                st.markdown(f"{row['description']}")
                st.markdown("---")
        else:
            st.error("No books found! Try a different title.")
