import streamlit as st 

st.set_page_config(
    page_title="Spotify Music Recommendation System",
    page_icon="🎵"
)

st.header("🎵 Spotify Music Recommendation System")

st.subheader("Objective Of This Project")
st.write("""Develop a recommendation algorithm that uses the Spotify dataset to suggest music based on user preferences and song attributes.

The objective will be achieved by:""")

st.subheader("1. Analyzing four datasets:")
st.write("""
- Data dataset
- Genre dataset
- Artist dataset
- Year dataset
""")

st.subheader("2. Identifying songs and trends:")
st.write("""
- Exploring song features over the years
- Highlighting popular songs across different years
- Identifying the most popular genres
- Finding the most listened-to artists
""")

st.subheader("3. Contribution and Goals:")
st.write("""
- Improve music recommendation accuracy by at least 15–20% compared to a baseline model
- Increase user engagement (e.g., average listening time) by 10%
- Provide personalized recommendations that adapt to user preferences over time
- Deliver insights into evolving music trends that can inform playlist generation and marketing strategies
""")

st.subheader("Data Understanding")
st.write("""
The main datasets I am working with are:

1. **Data dataset** - Core track features and metadata
2. **Artist dataset** - Links tracks to performers and artist information
3. **Genre dataset** - Categorizes tracks for genre-based filtering
4. **Year dataset** - Enables trend analysis and era-based recommendations
  
These datasets provide comprehensive information to support building a music recommendation system using Spotify data.

**Dataset Relationships:**
- The Data dataset contains individual track features and metadata
- The Artist dataset connects tracks to their performers
- The Genre dataset enables genre-based categorization and filtering
- The Year dataset allows temporal analysis and trend identification
- Together, they enable both micro-level (individual track features) and macro-level (artist popularity, genre trends) analysis
""")

st.subheader("Key Features")
st.write("""
1. **Valence**: Measures the musical positiveness or emotional quality of a track on a scale from 0.0 (sad/negative) to 1.0 (happy/positive).

2. **Year**: The year the music was released.

3. **Acousticness**: Indicates the likelihood that a track is acoustic (non-electric instruments vs. electronic/amplified sounds). A higher score (closer to 1.0) means the track is more acoustic, while a lower score suggests greater reliance on synthesized sounds.

4. **Artist**: The creator or performer of the music.

5. **Danceability**: Describes how suitable a track is for dancing. A score of 0.0 represents the least danceable music, while a score of 1.0 indicates a highly danceable track.

6. **Duration_ms**: The length of the track in milliseconds.

7. **Energy**: Represents the intensity and activity level of a track, measured from 0.0 to 1.0.

8. **Explicit**: Indicates whether the track contains content unsuitable for children (e.g., strong language, violence, sexual content, or drug references).

9. **Id**: A unique identifier for the track.

10. **Instrumentalness**: Predicts whether a track has no vocals. Values closer to 1.0 suggest a higher probability of being purely instrumental.

11. **Key**: The musical key of the track (e.g., C, C#, D, etc.).

12. **Liveness**: Estimates the likelihood that a track was performed live, based on audience noise and presence. Higher values indicate a greater probability of live performance.

13. **Loudness**: The overall volume of the track, measured in decibels (dB). Typically ranges from -60 to 0 dB.

14. **Mode**: Refers to the modality of the track, typically distinguishing between major (1 - tends to sound positive/happy) and minor (0 - tends to sound negative/sad).

15. **Name**: The title of the track.

16. **Popularity**: A measure of the track's current popularity (0-100), calculated by algorithm based on total plays and how recent those plays are.

17. **Release_date**: The date the track was released.

18. **Speechiness**: Detects the presence of spoken words in a track (0.0 to 1.0). Values above 0.66 indicate tracks made entirely of spoken words (podcasts, audiobooks). Values between 0.33-0.66 may contain both music and speech. Below 0.33 represents mostly music.

19. **Tempo**: The speed or pace of a track, measured in beats per minute (BPM).

20. **Genres**: Categories or classifications that group tracks with similar musical characteristics (e.g., pop, rock, hip-hop, jazz).
""")

st.subheader("Understanding the Context of the Data")
st.write("""
Music recommendation systems are an essential feature of modern streaming platforms such as Spotify. With millions of tracks available, users often rely on intelligent algorithms to discover music that fits their tastes and moods.

The purpose of this project is to build a recommendation algorithm that uses Spotify's datasets to suggest music based on user preferences and song attributes. Unlike simple playlists, recommendation systems analyze both user listening behavior (favorite songs, artists, and genres) and track-level features (e.g., tempo, energy, valence, acousticness) to deliver personalized suggestions.
""")

st.subheader("Recommendation Strategy")
st.write("""
This system will employ multiple approaches:

**Content-based Filtering:**
- Uses audio features (tempo, energy, valence, acousticness, danceability) to find similar tracks
- Recommends songs with similar characteristics to those a user already enjoys

**Collaborative Filtering:**
- Analyzes user behavior patterns to suggest tracks liked by similar users
- Identifies patterns in listening habits across the user base

**Hybrid Approach:**
- Combines both content-based and collaborative methods for improved accuracy
- Leverages strengths of both approaches while mitigating their weaknesses
""")

st.subheader("Expected Outcomes")
st.write("""
By studying listening patterns, track characteristics, and trends across years, this project will:

1. **Help users discover new music** aligned with their preferences
2. **Adapt to evolving user tastes** over time through continuous learning
3. **Provide insights into broader music trends** such as genre shifts, artist popularity, and emerging styles
4. **Reduce search time** by proactively suggesting relevant tracks
5. **Increase discovery of niche/lesser-known artists** to avoid filter bubbles
6. **Enable context-aware recommendations** based on mood, time of day, or activity

Ultimately, the system aims to improve the user experience, boost engagement metrics (listening time, session length, track completion rate), and enhance music discovery on streaming platforms.
""")

st.subheader("Success Metrics")
st.write("""
The system's effectiveness will be measured by:

1. **Recommendation Accuracy**: Precision, recall, and F1-score of suggestions
2. **User Engagement**: Click-through rate, listening completion rate, and session duration
3. **Diversity**: Variety of genres and artists recommended to avoid echo chambers
4. **Coverage**: Ability to recommend across the entire catalog, including long-tail content
5. **Novelty**: Balance between familiar and new discoveries
""")

st.subheader("Challenges to Address")
st.write("""
1. **Data Sparsity**: Not all users rate or listen to all songs
2. **Scalability**: Processing millions of tracks and users efficiently
3. **Popularity Bias**: Avoiding over-recommendation of already popular tracks
4. **Cold Start Problem**: Handling new users and new tracks with limited data
5. **Privacy**: Handling user data responsibly and transparently
6. **Real-time Updates**: Adapting to changing user preferences quickly
""")

st.markdown("---")
st.caption("This recommendation system leverages machine learning and data analysis to create personalized music experiences.")
