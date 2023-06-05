import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MovieRecommender:
    def __init__(self, data):
        self.data = data
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.data['overview'] = self.data['overview'].fillna('')
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['overview'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()

    def recommend_movies(self, title, num_recommendations=10):
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(num_recommendations+1)]
        movie_indices = [i[0] for i in sim_scores]
        return self.data['title'].iloc[movie_indices]

if __name__ == "__main__":
    data = pd.read_csv('movies_metadata.csv')  # your path to the dataset
    recommender = MovieRecommender(data)
    print(recommender.recommend_movies('Toy Story'))
