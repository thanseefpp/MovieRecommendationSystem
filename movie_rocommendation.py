import pandas as pd
import difflib # using to get the close matches
from sklearn.feature_extraction.text import TfidfVectorizer # to transform the text data to numerical feature vector.
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Loading the dataset using Pandas dataframe
movies_data = pd.read_csv('Data/movies.csv')

# selecting the relevant features for the recommendation
selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null values with null string to avoid error.

for features in selected_features:
  movies_data[features] = movies_data[features].fillna("")
  
# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# converting the text data to numerical data
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity score using the cosine similarity
similarity = cosine_similarity(feature_vectors)

st.title("Movie Recommendation System")

movie_name = st.text_input("Enter your favorite movie name : ")

# movie_name = input("Enter your favorite movie name : ")

list_all_movie_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name,list_all_movie_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similarity_movies = sorted(similarity_score, key= lambda x:x[1], reverse=True)

print("Suggested Movies for you :\n")

i = 1

for movie in sorted_similarity_movies:
  index = movie[0] #to get the index value
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<=20):
    print(f"{i} -> {title_from_index}")
    i+=1