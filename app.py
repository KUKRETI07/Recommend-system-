import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset from CSV
try:
    new_df = pd.read_csv('new_df.csv')
except FileNotFoundError:
    st.error("❌ The file 'new_df.csv' was not found. Please upload it to your repository.")
    st.stop()


# Preprocess and vectorize
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_df['Tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(product_name, cosine_sim=cosine_sim):
    idx = new_df[new_df['product_name'] == product_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    product_indices = [i[0] for i in sim_scores]
    return new_df['product_name'].iloc[product_indices]

# Streamlit UI
st.title("🛒 Product Recommender")
selected_product = st.selectbox("Choose a product:", new_df['product_name'].values)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_product)
    st.subheader("Top 5 Recommendations:")
    for i, rec in enumerate(recommendations, start=1):
        st.write(f"{i}. {rec}")
