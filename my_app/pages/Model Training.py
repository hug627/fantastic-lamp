import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Model Training", layout="wide")
st.title("🤖 Model Training")
st.subheader("Spotify Music Recommendation System")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\\Users\\Nullvoid\Desktop\\Music recommendation system\\Spotify\\data.csv")


data = load_data()

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
               'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

scaler = StandardScaler()
scaler.fit(data[number_cols])

# ===============================
# SPOTIFY API CONFIG
# ===============================
os.environ["SPOTIPY_CLIENT_ID"] = "fdc9a50a91db41a3a25dcb6c76ebc29d"
os.environ["SPOTIPY_CLIENT_SECRET"] = "9294659f5123413d94321a6506f68ffe"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ["SPOTIPY_CLIENT_ID"],
    client_secret=os.environ["SPOTIPY_CLIENT_SECRET"]
))

# ===============================
# HELPER FUNCTIONS
# ===============================
def get_song_data(song, data):
    filtered_data = data[(data['name'] == song['name']) & (data['year'] == song['year'])]
    if not filtered_data.empty:
        return filtered_data.iloc[0]
    query = f"track:{song['name']} year:{song['year']}"
    results = sp.search(q=query, limit=1, type="track")
    if not results["tracks"]["items"]:
        return None
    track = results["tracks"]["items"][0]
    track_id = track["id"]
    audio_features = sp.audio_features(track_id)[0]
    if audio_features is None:
        return None
    song_data = {
        "name": track["name"],
        "year": int(track["album"]["release_date"].split("-")[0]),
        "artists": ", ".join([artist["name"] for artist in track["artists"]]),
        "explicit": int(track["explicit"]),
        "duration_ms": track["duration_ms"],
        "popularity": track["popularity"],
    }
    song_data.update(audio_features)
    return pd.Series(song_data)

def get_mean_vector(song_list, data):
    vectors = []
    for song in song_list:
        song_data = get_song_data(song, data)
        if song_data is not None:
            vectors.append(song_data[number_cols].values)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def recommend_songs(song_list, data, n_songs=10):
    song_center = get_mean_vector(song_list, data)
    if song_center is None:
        return []
    scaled_data = scaler.transform(data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin([s['name'] for s in song_list])]
    return rec_songs[['name', 'year', 'artists', 'popularity']]

# ===============================
# STREAMLIT UI
# ===============================

st.markdown("Enter your favorite songs (name + year) and get personalized recommendations!")

with st.form("song_input"):
    song1 = st.text_input("Song 1 Name", "Beat It")
    year1 = st.number_input("Year 1", min_value=1900, max_value=2025, value=1982)
    song2 = st.text_input("Song 2 Name", "Billie Jean")
    year2 = st.number_input("Year 2", min_value=1900, max_value=2025, value=1982)
    song3 = st.text_input("Song 3 Name", "Thriller")
    year3 = st.number_input("Year 3", min_value=1900, max_value=2025, value=1982)
    submit = st.form_submit_button("Get Recommendations")

if submit:
    song_list = []
    if song1: song_list.append({'name': song1, 'year': year1})
    if song2: song_list.append({'name': song2, 'year': year2})
    if song3: song_list.append({'name': song3, 'year': year3})

    if song_list:
        recs = recommend_songs(song_list, data)
        if recs is None or recs.empty:
            st.warning("❌ No recommendations found. Try different songs.")
        else:
            st.success(f"Found {len(recs)} recommendations!")
            st.dataframe(recs)
            fig = px.bar(recs, x="name", y="popularity", color="artists",
                         title="Recommended Songs Popularity", labels={"name":"Song"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter at least one song.")

st.markdown("---")
st.write("Thank you for using the Spotify Music Recommendation System!")
st.caption("Built with ❤️ using Spotify API + Streamlit")
