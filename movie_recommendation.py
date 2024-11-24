
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("movies.csv")

# Split the Genre column into multiple binary columns
df['Genre'] = df['Genre'].str.split(", ")

# Apply MultiLabelBinarizer to the Genre column
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['Genre'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

df = pd.concat([df, genre_df], axis=1)

# Frequency encoding for director
df['director_encoded'] = df['Director'].map(df['Director'].value_counts())

# Mean imputation for Metascore and Revenue
df['Metascore'] = df['Metascore'].fillna(df['Metascore'].mean())
df['Revenue'] = df['Revenue'].fillna(df['Revenue'].mean())

# Drop rows where Director column is null
df = df.dropna(subset=['Director'])

df = df.drop(columns=["Genre", "Director", "Description", "Rank"])

# Moving Title to the last column in the DataFrame
title_column = df.pop('Title')
df['Title'] = title_column

def euclidean_distance(sample1, sample2):
    return np.sqrt(np.sum((sample1 - sample2) ** 2))

def knn(train_data, train_labels, test_point, k=3):
    # Calculate the distance from the test point to each training data point
    distances = []
    for i, data_point in enumerate(train_data):
        distance = euclidean_distance(test_point, data_point)
        distances.append((distance, train_labels[i]))
    
    distances.sort(key=lambda x: x[0])

    k_nearest_labels = [label for _, label in distances[:k]]
    
    return k_nearest_labels

st.title("Movie Recommender")
st.subheader("This system will recommend 4 similar movies.")
movie_title = st.text_input("Enter the title of a movie you have watched.")

if movie_title:
    if movie_title in df["Title"].values:
        input_movie = df[df["Title"] == movie_title].iloc[0]
        input_features = input_movie.drop("Title").values.astype(np.float64)

        train_data = df[df["Title"] != movie_title].drop(columns=["Title"]).values.astype(np.float64)

        movie_titles = df[df["Title"] != movie_title]["Title"].values
        k = 4

        # Get the recommended movies
        recommended_movies = knn(train_data, movie_titles, input_features, k)

        # Display the recommended movies
        st.write(f"Recommended movies based on '{movie_title}':")
        for movie in recommended_movies:
            st.write(movie)
    else:
        st.write("Movie not found in the dataset.")
