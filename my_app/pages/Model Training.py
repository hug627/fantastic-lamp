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
# DOCUMENTATION SECTION
# ===============================
with st.expander("📖 How This Recommendation System Works"):
    st.markdown("""
    ### Algorithm Overview
    This system uses **Content-Based Filtering** with **Cosine Similarity** to recommend songs.
    
    **Step-by-Step Process:**
    
    1. **Feature Extraction**
       - Takes 15 numerical features from each song (valence, energy, danceability, etc.)
       - These features capture the "personality" of each track
    
    2. **Standardization**
       - Uses StandardScaler to normalize all features to the same scale
       - Prevents features with larger values (like duration_ms) from dominating the calculation
    
    3. **Mean Vector Calculation**
       - Calculates the average feature values of your input songs
       - This creates a "profile" of your musical taste
    
    4. **Similarity Measurement**
       - Uses Cosine Distance to measure how similar each song in the database is to your taste profile
       - Lower distance = more similar = better recommendation
    
    5. **Ranking and Selection**
       - Sorts all songs by similarity
       - Returns the top N most similar songs (excluding your input songs)
    
    ### Why Cosine Similarity?
    - Works well with high-dimensional data (15 features)
    - Measures angle between vectors, not just distance
    - Effective for capturing musical similarity patterns
    
    ### Features Used in Recommendations:
    - **Acoustic properties**: acousticness, instrumentalness
    - **Energy metrics**: energy, loudness, tempo
    - **Mood indicators**: valence (happiness), mode (major/minor)
    - **Rhythm features**: danceability, speechiness
    - **Contextual data**: popularity, year, explicit content
    """)

with st.expander("🎯 Expected Results & Interpretation"):
    st.markdown("""
    ### What You'll See:
    
    **Recommendation Table:**
    - Song names with their release years
    - Artist information
    - Popularity scores (0-100, higher = more popular)
    
    **Popularity Chart:**
    - Bar chart showing recommended songs ranked by popularity
    - Color-coded by artist for easy identification
    - Helps you see both similar AND popular tracks
    
    ### How to Interpret Results:
    
    **High Similarity Songs:**
    - Share similar audio features with your input
    - May be from same genre or era
    - Good for "more of the same" discovery
    
    **Diverse Recommendations:**
    - System may suggest songs from different genres if they share key features
    - Example: A calm acoustic pop song might match with jazz if both have high acousticness
    
    **Year Considerations:**
    - Model considers year as a feature, so recommendations may cluster around similar time periods
    - To get modern versions of classic sounds, try mixing input song years
    
    ### Typical Patterns:
    - **Input: Upbeat pop songs** → Recommendations: High danceability, high energy tracks
    - **Input: Acoustic ballads** → Recommendations: Low energy, high acousticness, positive valence
    - **Input: Electronic/EDM** → Recommendations: High energy, low acousticness, high tempo
    """)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join('Spotify/data.csv')
    
    try:
        df = pd.read_csv(file_path)
        st.sidebar.success(f"✅ Loaded {len(df):,} songs from database")
        return df
    except FileNotFoundError:
        st.error(f"❌ Could not find data.csv at: {file_path}")
        st.info("Please ensure data.csv is in your repository at the correct location.")
        st.stop()

data = load_data()

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
               'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

scaler = StandardScaler()
scaler.fit(data[number_cols])

# ===============================
# SPOTIFY API CONFIG
# ===============================
os.environ["SPOTIPY_CLIENT_ID"] = st.secrets.get("SPOTIPY_CLIENT_ID", "fdc9a50a91db41a3a25dcb6c76ebc29d")
os.environ["SPOTIPY_CLIENT_SECRET"] = st.secrets.get("SPOTIPY_CLIENT_SECRET", "9294659f5123413d94321a6506f68ffe")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ["SPOTIPY_CLIENT_ID"],
    client_secret=os.environ["SPOTIPY_CLIENT_SECRET"]
))

# ===============================
# HELPER FUNCTIONS
# ===============================
def get_song_data(song, data):
    """
    Retrieves song data either from local dataset or Spotify API.
    
    Process:
    1. First searches local dataset for exact match (name + year)
    2. If not found, queries Spotify API for real-time data
    3. Extracts audio features and metadata
    
    Returns: pandas Series with song features or None if not found
    """
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
    """
    Calculates the average feature vector from input songs.
    
    This creates a "taste profile" by averaging the numerical features
    of all input songs. This profile represents the user's preferences.
    
    Returns: numpy array of averaged features or None if no valid songs
    """
    vectors = []
    for song in song_list:
        song_data = get_song_data(song, data)
        if song_data is not None:
            vectors.append(song_data[number_cols].values)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def recommend_songs(song_list, data, n_songs=10):
    """
    Main recommendation function using cosine similarity.
    
    Algorithm:
    1. Calculate mean vector from input songs (taste profile)
    2. Standardize all features using pre-fitted scaler
    3. Calculate cosine distance between taste profile and all songs
    4. Sort by distance (lower = more similar)
    5. Return top N songs, excluding input songs
    
    Returns: DataFrame with recommended songs or None if error
    """
    song_center = get_mean_vector(song_list, data)
    if song_center is None:
        return None
    
    scaled_data = scaler.transform(data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = data.iloc[index].copy()
    rec_songs = rec_songs[~rec_songs['name'].isin([s['name'] for s in song_list])]
    
    required_cols = ['name', 'year', 'popularity']
    if 'artists' not in rec_songs.columns:
        rec_songs['artists'] = 'Unknown'
    
    return rec_songs[required_cols + ['artists']]

# ===============================
# STREAMLIT UI
# ===============================

st.markdown("### 🎵 Enter Your Favorite Songs")
st.markdown("Provide at least one song to get personalized recommendations based on audio features and similarity.")

with st.form("song_input"):
    col1, col2 = st.columns(2)
    
    with col1:
        song1 = st.text_input("Song 1 Name", "Beat It", help="Enter exact song title")
        song2 = st.text_input("Song 2 Name", "Billie Jean")
        song3 = st.text_input("Song 3 Name", "Thriller")
    
    with col2:
        year1 = st.number_input("Year 1", min_value=1900, max_value=2025, value=1982)
        year2 = st.number_input("Year 2", min_value=1900, max_value=2025, value=1982)
        year3 = st.number_input("Year 3", min_value=1900, max_value=2025, value=1982)
    
    submit = st.form_submit_button("🚀 Get Recommendations", type="primary")

if submit:
    song_list = []
    if song1: song_list.append({'name': song1, 'year': year1})
    if song2: song_list.append({'name': song2, 'year': year2})
    if song3: song_list.append({'name': song3, 'year': year3})

    if song_list:
        with st.spinner("🎵 Analyzing your taste profile and finding recommendations..."):
            recs = recommend_songs(song_list, data)
        
        if recs is None or len(recs) == 0:
            st.warning("❌ No recommendations found. Try different songs or check spelling.")
        else:
            st.success(f"✅ Found {len(recs)} recommendations based on your taste!")
            
            # Display results summary
            st.markdown("### 📊 Your Recommendations")
            st.dataframe(recs, use_container_width=True)
            
            # Analysis summary
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_pop = recs['popularity'].mean()
                st.metric("Average Popularity", f"{avg_pop:.1f}/100")
            with col2:
                year_range = f"{recs['year'].min()} - {recs['year'].max()}"
                st.metric("Year Range", year_range)
            with col3:
                unique_artists = recs['artists'].nunique()
                st.metric("Unique Artists", unique_artists)
            
            # Visualization
            try:
                if 'popularity' in recs.columns and not recs.empty:
                    fig = px.bar(
                        recs, 
                        x="name", 
                        y="popularity", 
                        color="artists",
                        title="Recommended Songs by Popularity", 
                        labels={"name": "Song", "popularity": "Popularity Score"},
                        height=400
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Chart Interpretation:**
                    - Taller bars = more popular/mainstream tracks
                    - Different colors = different artists
                    - Songs are ordered by similarity to your input (left = most similar)
                    """)
            except Exception as e:
                st.error(f"Could not create chart: {str(e)}")
            
            # Additional insights
            with st.expander("🔍 Analysis of Your Recommendations"):
                st.markdown(f"""
                **Summary Statistics:**
                - Total recommendations: {len(recs)}
                - Most popular song: **{recs.loc[recs['popularity'].idxmax(), 'name']}** ({recs['popularity'].max()}/100)
                - Least popular song: **{recs.loc[recs['popularity'].idxmin(), 'name']}** ({recs['popularity'].min()}/100)
                - Most represented artist: **{recs['artists'].mode()[0] if not recs['artists'].mode().empty else 'Various'}**
                
                **What This Means:**
                The algorithm found these songs by matching audio features (energy, danceability, mood, etc.) 
                with your input songs. Songs appear in order of similarity - the first recommendations are 
                closest matches to your taste profile.
                
                **Tips:**
                - If results are too similar, try mixing different genres in your input
                - If results are too diverse, use songs from the same era or genre
                - Popular songs appear more often in recommendations due to data availability
                """)
    else:
        st.warning("⚠️ Please enter at least one song to get recommendations.")

# Sidebar with technical details
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Technical Details")
st.sidebar.markdown(f"""
**Model Type:** Content-Based Filtering  
**Similarity Metric:** Cosine Distance  
**Features Used:** {len(number_cols)} numerical features  
**Database Size:** {len(data):,} songs
""")

st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.markdown("""
**Strengths:**
- Works without user history
- Provides explainable results
- Fast computation
- No cold start problem

**Limitations:**
- May suggest similar-sounding songs
- Requires accurate song metadata
- Cannot capture cultural context
- Limited by database size
""")

st.markdown("---")
st.write("Thank you for using the Spotify Music Recommendation System!")
st.caption("Built with ❤️ using Spotify API + Streamlit | Algorithm: Content-Based Filtering with Cosine Similarity")
