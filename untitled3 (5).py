import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.metrics import mean_squared_error
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

df = pd.read_csv('amazon.csv')

df.info()

df.head(3)

df['rating_count'].isnull().sum()

df['rating_count'] = df['rating_count'].replace({',': ''}, regex=True)
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
median_rating_count = df['rating_count'].median()
df['rating_count'] = df['rating_count'].fillna(median_rating_count)
df['rating_count'] = df['rating_count'].astype(int)

df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
df['discount_percentage'] = df['discount_percentage'].replace({'%': '', ',': ''}, regex=True).astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

df['rating_count'].isnull().sum()

df['category'] = df['category'].astype(str)

def preprocess_text(text):
    text = text.lower()
    return text

df['product_name'] = df['product_name'].apply(preprocess_text)
df['about_product'] = df['about_product'].apply(preprocess_text)

mlb = MultiLabelBinarizer()
df['category'] = df['category'].apply(lambda x: x.split('|'))
category_binarized = mlb.fit_transform(df['category'])
category_df = pd.DataFrame(category_binarized, columns=mlb.classes_, index=df.index)
df = pd.concat([df, category_df], axis=1)
df = df.drop('category', axis=1)

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = df['rating_count'].replace({',': '', '[^\d.]': ''}, regex=True)
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
df['product_name_tfidf'] = list(tfidf_vectorizer.fit_transform(df['product_name']).toarray())
df['about_product_tfidf'] = list(tfidf_vectorizer.fit_transform(df['about_product']).toarray())

product_name_tfidf = np.array(df['product_name_tfidf'].tolist())
about_product_tfidf = np.array(df['about_product_tfidf'].tolist())

cosine_sim_name = cosine_similarity(product_name_tfidf)
cosine_sim_about = cosine_similarity(about_product_tfidf)

cosine_sim = 0.6 * cosine_sim_name + 0.4 * cosine_sim_about

sparse_category_binarized = csr_matrix(category_binarized)
category_sim = 1 - cdist(sparse_category_binarized.toarray(), sparse_category_binarized.toarray(), metric='jaccard')

def compute_combined_similarity(cosine_sim, category_sim, weight_cosine=0.5, weight_category=0.5):
    return weight_cosine * cosine_sim + weight_category * category_sim

param_grid = {
    'weight_cosine': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'weight_category': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

best_score = float('inf')
best_params = {}

for weight_cosine in param_grid['weight_cosine']:
    for weight_category in param_grid['weight_category']:
        combined_sim = compute_combined_similarity(cosine_sim, category_sim, weight_cosine, weight_category)
        score = np.mean(np.abs(combined_sim - cosine_sim))
        if score < best_score:
            best_score = score
            best_params = {'weight_cosine': weight_cosine, 'weight_category': weight_category}

print(f"Best Weight for Cosine Similarity: {best_params['weight_cosine']}")
print(f"Best Weight for Category Similarity: {best_params['weight_category']}")

final_sim = compute_combined_similarity(cosine_sim, category_sim, weight_cosine=best_params['weight_cosine'], weight_category=best_params['weight_category'])

def get_recommendations(product_id, final_sim, df, top_n=5):
    product_idx = df[df['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(final_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['product_id', 'product_name']]

recommended_products = get_recommendations('B0BQRJ3C47', final_sim, df, top_n=5)
print(recommended_products)

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

        predicted_rating = np.mean([
            user_ratings[user_ratings['product_id'] == df.iloc[i]['product_id']]['rating'].mean()
            for i in sim_indices if df.iloc[i]['product_id'] in user_ratings['product_id'].values
        ])

        if not np.isnan(predicted_rating):
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)

    if len(actual_ratings) == 0 or len(predicted_ratings) == 0:
        print("Not enough data to calculate RMSE")
        return None

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print(f"RMSE: {rmse}")
    return rmse

rmse_score = evaluate_rmse(df, final_sim, user_ratings)

df['user_id'] = df['user_id'].astype(str)
df['product_id'] = df['product_id'].astype(str)
df = df[['user_id', 'product_id', 'rating']]

reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD(n_factors=50, random_state=42)
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")

def recommend_products(user_id, model, df, n=5):
    all_products = df['product_id'].unique()
    user_rated_products = df[df['user_id'] == user_id]['product_id'].tolist()

    predictions = [model.predict(user_id, pid) for pid in all_products if pid not in user_rated_products]

    top_products = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return [(p.iid, p.est) for p in top_products]

user_id = df['user_id'].iloc[0]
recommendations = recommend_products(user_id, model, df, n=5)
print("Top Recommendations:", recommendations)
