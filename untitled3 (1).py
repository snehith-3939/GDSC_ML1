import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load your data
df = pd.read_csv('amazon.csv')

# Preprocessing: Clean and process data (This part can be similar to your existing code)
df['rating_count'] = df['rating_count'].replace({',': ''}, regex=True)
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
df['rating_count'] = df['rating_count'].fillna(df['rating_count'].median())
df['rating_count'] = df['rating_count'].astype(int)

df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
df['discount_percentage'] = df['discount_percentage'].replace({'%': '', ',': ''}, regex=True).astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

df['product_name'] = df['product_name'].apply(lambda x: x.lower())
df['about_product'] = df['about_product'].apply(lambda x: x.lower())

# Vectorization and Similarity Calculations
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
product_name_tfidf = np.array(tfidf_vectorizer.fit_transform(df['product_name']).toarray())
about_product_tfidf = np.array(tfidf_vectorizer.fit_transform(df['about_product']).toarray())

cosine_sim_name = cosine_similarity(product_name_tfidf)
cosine_sim_about = cosine_similarity(about_product_tfidf)
final_sim = 0.8 * cosine_sim_name + 0.2 * cosine_sim_about  # Example weighted similarity

# Streamlit App UI
st.title("Product Recommender System")

st.write("Welcome to the Product Recommender! This app recommends products based on your selection.")

# Display product selection dropdown
product_options = df['product_name'].unique()
selected_product = st.selectbox("Select a product to get recommendations:", product_options)

# Recommend top N products based on similarity
def get_recommendations(product_name, final_sim, df, top_n=5):
    product_idx = df[df['product_name'] == product_name].index[0]
    sim_scores = list(enumerate(final_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['product_name', 'discounted_price', 'rating']]

# Display recommendations
if selected_product:
    recommendations = get_recommendations(selected_product, final_sim, df)
    st.write(f"Top 5 Recommendations for '{selected_product}':")
    st.write(recommendations)

# RMSE Calculation (for evaluation)
user_ratings = df[['user_id', 'product_id', 'rating']].dropna()

def evaluate_rmse(df, final_sim, user_ratings):
    actual_ratings = []
    predicted_ratings = []

    for index, row in user_ratings.iterrows():
        product_id = row['product_id']
        actual_rating = row['rating']
        if product_id not in df['product_id'].values:
            continue
        product_idx = df[df['product_id'] == product_id].index[0]
        sim_scores = list(enumerate(final_sim[product_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        sim_indices = [i[0] for i in sim_scores]
        predicted_rating = np.mean([user_ratings[user_ratings['product_id'] == df.iloc[i]['product_id']]['rating'].mean() for i in sim_indices if df.iloc[i]['product_id'] in user_ratings['product_id'].values])
        if not np.isnan(predicted_rating):
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)
    
    if len(actual_ratings) == 0 or len(predicted_ratings) == 0:
        return None
    return np.sqrt(np.mean((np.array(actual_ratings) - np.array(predicted_ratings)) ** 2))

rmse_score = evaluate_rmse(df, final_sim, user_ratings)
st.write(f"RMSE Score for the recommender system: {rmse_score:.4f}")

# SVD model (Collaborative Filtering)
df['user_id'] = df['user_id'].astype(str)
df['product_id'] = df['product_id'].astype(str)

reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD(n_factors=50, random_state=42)
model.fit(trainset)

# Function for collaborative filtering recommendations
def recommend_products(user_id, model, df, n=5):
    all_products = df['product_id'].unique()
    user_rated_products = df[df['user_id'] == user_id]['product_id'].tolist()
    predictions = [model.predict(user_id, pid) for pid in all_products if pid not in user_rated_products]
    top_products = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return [(p.iid, p.est) for p in top_products]

# Display collaborative filtering recommendations
user_id = df['user_id'].iloc[0]  # Example user id
collab_recommendations = recommend_products(user_id, model, df, n=5)
st.write(f"Collaborative Filtering Recommendations for User {user_id}:")
st.write(collab_recommendations)
