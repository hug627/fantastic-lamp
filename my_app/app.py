import streamlit as st 

st.set_page_config(
    page_title="Spotify Music Recommendation System",
    page_icon="🎵 "
)
st.header("🎵Spotify Music Recommendation System")
st.subheader("Objective Of This Project")
st.write(""" Develop a recommendation algorithm that uses the Spotify dataset to suggest music based on user preferences and song attributes.

The objective will be achieved by:""")

st.subheader("1. Analyzing four datasets:")
st.write("""

i. Data dataset

ii. Genre dataset

iii. Artist dataset

iv. Year dataset""")

st.subheader("2. Identifying songs and trends:")
st.write("""
i. Exploring song features over the years

ii. Highlighting popular songs across different years

iii. Identifying the most popular genres

iv. Finding the most listened-to artists""")

st.subheader("3. Contribution and Goals:")
st.write("""
i. Improve music recommendation accuracy by at least 15–20% compared to a baseline model

ii. Increase user engagement (e.g., average listening time) by 10%

iii. Provide personalized recommendations that adapt to user preferences over time

iv. Deliver insights into evolving music trends that can inform playlist generation and marketing strategies
""")
st.subheader(" Data Understanding:")
st.write("""
The main datasets I am working with are:

  1. Data dataset
  2. Artist dataset
  3. Genre dataset
  4. Year dataset
  
These datasets provide comprehensive information to support building a music recommendation system using Spotify data.
""")
st.subheader("Key Features:")
st.write("""
         
1. Valence: Measures the musical positiveness or emotional quality of a track, ranging from negative to positive. Typically represented on a scale from 0.0 to 1.0.
2. Year: The year the music was released.
3. Acousticness: Indicates the likelihood that a track is acoustic (non-electric instruments vs. electronic/amplified sounds). A higher score (closer to 1.0) means the track is more acoustic, while a lower score suggests greater reliance on synthesized sounds.
4. Artist: The creator or performer of the music.
5. Danceability: Describes how suitable a track is for dancing. A score of 0.0 represents the least danceable music, while a score of 1.0 indicates a highly danceable track.
6. Duration_ms: The length of the track in milliseconds.
7. Energy: Represents the intensity and activity level of a track.
8. Explicit: Indicates whether the track contains content unsuitable for children (e.g., strong language, violence, sexual content, or drug references).
9. Id: A unique identifier for the track.
10. Instrumentalness: Predicts whether a track has no vocals. Values closer to 1.0 suggest a higher probability of being purely instrumental.
11. Key: The group of pitches or scale of the track.
12. Liveness: Estimates the likelihood that a track was performed live, based on audience noise and presence.
13. Loudness: The overall volume of the track, measured in decibels (dB).
14. Mode: Refers to the modality of the track, typically distinguishing between major (positive, happy) and minor (negative, sad) tonalities.
15. Name: The title of the track.
16. Popularity: A measure of the track’s success, based on chart rankings, sales, radio play, and streaming activity.
17. Release_date: The date the track was released.
18. Speechiness: Detects the presence of spoken words in a track. Higher values indicate more speech-like recordings.
19. Tempo: The speed or pace of a track, measured in beats per minute (BPM).
20. Count: Represents beat tracking or the number of times beats are verbally counted to follow rhythm.
21. Genres: Categories or classifications that group tracks with similar musical characteristics.
""")
st.subheader("Understanding the context of the data")
st.write("""Music recommendation systems are an essential feature of modern streaming platforms such as Spotify. With millions of tracks available, users often rely on intelligent algorithms to discover music that fits their tastes and moods.

The purpose of this project is to build a recommendation algorithm that uses Spotify’s datasets to suggest music based on user preferences and song attributes. Unlike simple playlists, recommendation systems analyze both user listening behavior (favorite songs, artists, and genres) and track-level features (e.g., tempo, energy, valence, acousticness) to deliver personalized suggestions.

By studying listening patterns, track characteristics, and trends across years, this project will:

1. Help users discover new music aligned with their preferences.
2. Adapt to evolving user tastes over time.
3. Provide insights into broader music trends such as genre shifts or artist popularity.

Ultimately, the system aims to improve the user experience, boost engagement, and enhance music discovery on Spotify.""")