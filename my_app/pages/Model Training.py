import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# ===============================
# PAGE CONFIG & STYLING
# ===============================
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark base */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Main title */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(135deg, #1DB954 0%, #4af584 50%, #a8edbc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #e8e8f0 !important;
}

/* Cards */
.stat-card {
    background: linear-gradient(135deg, #131320 0%, #1a1a2e 100%);
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.3s ease;
}
.stat-card:hover { border-color: #1DB954; }
.stat-card .label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 6px;
}
.stat-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #1DB954;
}

/* Song input cards */
.song-input-wrapper {
    background: #131320;
    border: 1px solid #2a2a45;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}

/* Genre pill */
.genre-pill {
    display: inline-block;
    background: rgba(29, 185, 84, 0.15);
    border: 1px solid rgba(29, 185, 84, 0.4);
    color: #1DB954;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
    letter-spacing: 0.04em;
}

/* Source badge */
.badge-local {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    border-radius: 50px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 500;
}
.badge-api {
    background: rgba(29, 185, 84, 0.15);
    border: 1px solid rgba(29, 185, 84, 0.4);
    color: #4af584;
    border-radius: 50px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 500;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d0d18 !important;
    border-right: 1px solid #1e1e35 !important;
}

/* Streamlit form & inputs */
div[data-testid="stForm"] {
    background: #131320;
    border: 1px solid #2a2a45;
    border-radius: 18px;
    padding: 1.8rem;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0a0a0f !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #1DB954 !important;
    box-shadow: 0 0 0 2px rgba(29,185,84,0.2) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1DB954, #17a349) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 2.2rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(29,185,84,0.35) !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #2a2a45 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #131320 !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #131320;
    border: 1px solid #2a2a45;
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #6b6b8a !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #1DB954 !important; font-family: 'Syne', sans-serif !important; }

/* Info / warning / success */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

/* Divider */
hr { border-color: #1e1e35 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1DB954; }
</style>
""", unsafe_allow_html=True)


# ===============================
# CONSTANTS
# ===============================
NUMBER_COLS = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
    'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
]

FEATURE_DESCRIPTIONS = {
    'valence':          ('😊 Valence',        'Musical positivity (0=sad, 1=happy)'),
    'energy':           ('⚡ Energy',          'Intensity and activity level'),
    'danceability':     ('💃 Danceability',    'How suitable for dancing'),
    'acousticness':     ('🎸 Acousticness',    'Likelihood of being acoustic'),
    'instrumentalness': ('🎻 Instrumentalness','Predicts no vocal content'),
    'speechiness':      ('🗣️ Speechiness',     'Presence of spoken words'),
    'liveness':         ('🎤 Liveness',        'Presence of live audience'),
    'tempo':            ('🥁 Tempo',           'Beats per minute'),
}


# ===============================
# DATA LOADING
# ===============================
@st.cache_data
def load_data():
    file_path = os.path.join('Spotify', 'data.csv')
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("❌ Could not find `Spotify/data.csv`. Please ensure the file exists in your repository.")
        st.stop()

data = load_data()

# Fit scaler once
scaler = StandardScaler()
scaler.fit(data[NUMBER_COLS])


# ===============================
# SPOTIFY API SETUP
# ===============================
@st.cache_resource
def get_spotify_client():
    client_id     = st.secrets.get("SPOTIPY_CLIENT_ID",     "f0cbf773641d4956a3b08f68af0c5aea")
    client_secret = st.secrets.get("SPOTIPY_CLIENT_SECRET", "6c6fce81a5b94b0abf99bb597c2bab6f")
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    ))

sp = get_spotify_client()


# ===============================
# CORE FUNCTIONS
# ===============================

def fetch_from_spotify(song_name: str, year: int | None = None) -> pd.Series | None:
    """
    Query Spotify API for a song's audio features and metadata.
    Falls back to a name-only search if year-constrained search returns nothing.

    Returns a pd.Series with all NUMBER_COLS fields + name/artists/popularity/genre,
    or None if the track / its audio features cannot be found.
    """
    query = f"track:{song_name} year:{year}" if year else f"track:{song_name}"
    results = sp.search(q=query, limit=1, type="track")

    # Fallback: drop year constraint
    if not results["tracks"]["items"] and year:
        results = sp.search(q=f"track:{song_name}", limit=1, type="track")

    if not results["tracks"]["items"]:
        return None

    track          = results["tracks"]["items"][0]
    track_id       = track["id"]
    audio_features = sp.audio_features(track_id)[0]

    if audio_features is None:
        return None

    # Fetch genre from the primary artist
    artist_id  = track["artists"][0]["id"]
    artist_obj = sp.artist(artist_id)
    genres     = artist_obj.get("genres", [])

    song_data = {
        "name":       track["name"],
        "year":       int(track["album"]["release_date"].split("-")[0]),
        "artists":    ", ".join(a["name"] for a in track["artists"]),
        "explicit":   int(track["explicit"]),
        "duration_ms": track["duration_ms"],
        "popularity": track["popularity"],
        "genre":      genres[0] if genres else "unknown",
        "source":     "spotify_api",
    }
    song_data.update(audio_features)
    return pd.Series(song_data)


def get_song_data(song: dict, df: pd.DataFrame) -> pd.Series | None:
    """
    Retrieve song features. Checks the local dataset first; falls back to Spotify API.
    Attaches a 'source' tag ('dataset' or 'spotify_api').
    """
    match = df[(df['name'].str.lower() == song['name'].lower()) &
               (df['year'] == song['year'])]
    if not match.empty:
        result         = match.iloc[0].copy()
        result['source'] = 'dataset'
        if 'genre' not in result.index:
            result['genre'] = 'N/A'
        return result

    # Not in dataset → try Spotify
    return fetch_from_spotify(song['name'], song.get('year'))


def get_mean_vector(song_list: list[dict], df: pd.DataFrame):
    """
    Build a taste-profile vector by averaging the numerical features
    of all successfully resolved input songs.

    Returns (np.ndarray, list[dict]) — the mean vector and enriched song info.
    """
    vectors    = []
    song_info  = []

    for song in song_list:
        data_row = get_song_data(song, df)
        if data_row is not None:
            # Ensure all NUMBER_COLS are numeric and present
            try:
                vec = data_row[NUMBER_COLS].astype(float).values
                vectors.append(vec)
                song_info.append({
                    'name':    data_row.get('name',    song['name']),
                    'artists': data_row.get('artists', 'Unknown'),
                    'year':    data_row.get('year',    song['year']),
                    'genre':   data_row.get('genre',   'unknown'),
                    'source':  data_row.get('source',  'unknown'),
                })
            except (KeyError, ValueError):
                pass

    if not vectors:
        return None, []

    return np.mean(vectors, axis=0), song_info


def recommend_songs(song_list: list[dict], df: pd.DataFrame, n_songs: int = 10):
    """
    Content-based filtering using cosine similarity.

    Steps:
      1. Build mean feature vector from inputs (taste profile)
      2. Standardise all features with the pre-fitted scaler
      3. Calculate cosine distance to every song in the dataset
      4. Return top-N closest songs (excluding the input songs themselves)

    Returns (DataFrame of recommendations, list of enriched input song info).
    """
    mean_vec, enriched_inputs = get_mean_vector(song_list, df)

    if mean_vec is None:
        return None, []

    scaled_data   = scaler.transform(df[NUMBER_COLS])
    scaled_center = scaler.transform(mean_vec.reshape(1, -1))
    distances     = cdist(scaled_center, scaled_data, 'cosine')[0]

    # Sort and take top candidates (extra buffer for filtering)
    top_indices = np.argsort(distances)[:n_songs * 3]
    recs        = df.iloc[top_indices].copy()
    recs['similarity_score'] = 1 - distances[top_indices]  # higher = more similar

    # Exclude input songs
    input_names_lower = {s['name'].lower() for s in song_list}
    recs = recs[~recs['name'].str.lower().isin(input_names_lower)]
    recs = recs.head(n_songs).reset_index(drop=True)

    if 'artists' not in recs.columns:
        recs['artists'] = 'Unknown'
    if 'genre' not in recs.columns:
        recs['genre'] = 'N/A'

    recs['rank'] = range(1, len(recs) + 1)
    return recs[['rank', 'name', 'year', 'artists', 'genre', 'popularity', 'similarity_score']], enriched_inputs


def build_audio_radar(song_data_row: pd.Series, title: str) -> go.Figure:
    """Radar chart for a song's audio features."""
    features = ['valence', 'energy', 'danceability', 'acousticness',
                'instrumentalness', 'speechiness', 'liveness']
    values   = [float(song_data_row.get(f, 0)) for f in features]
    labels   = [FEATURE_DESCRIPTIONS[f][0] for f in features]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        line=dict(color='#1DB954', width=2),
        fillcolor='rgba(29,185,84,0.15)',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#0d0d18',
            radialaxis=dict(visible=True, range=[0, 1], color='#3a3a5c',
                            tickfont=dict(color='#6b6b8a', size=9)),
            angularaxis=dict(color='#3a3a5c',
                             tickfont=dict(color='#a0a0c0', size=10)),
        ),
        paper_bgcolor='#131320',
        plot_bgcolor='#131320',
        title=dict(text=title, font=dict(color='#e8e8f0', family='Syne', size=13)),
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
        height=300,
    )
    return fig


# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
    st.markdown("---")

    st.markdown("### 📊 Database Stats")
    st.markdown(f"""
    <div class="stat-card" style="margin-bottom:10px">
        <div class="label">Songs in Dataset</div>
        <div class="value">{len(data):,}</div>
    </div>
    <div class="stat-card" style="margin-bottom:10px">
        <div class="label">Features Used</div>
        <div class="value">{len(NUMBER_COLS)}</div>
    </div>
    <div class="stat-card">
        <div class="label">Year Range</div>
        <div class="value">{int(data['year'].min())}–{int(data['year'].max())}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 🧠 Algorithm
    **Content-Based Filtering** with **Cosine Similarity**

    1. Extracts 15 audio features per song
    2. Standardises to zero mean / unit variance
    3. Averages inputs → taste profile
    4. Ranks all songs by cosine distance
    5. Returns the closest N matches

    Songs not in the dataset are fetched **live from Spotify API**, so any genre works.
    """)

    st.markdown("---")
    st.caption("Built with Streamlit · Spotify API · scikit-learn")


# ===============================
# MAIN PAGE
# ===============================
st.title("🎵 Music Recommender")
st.markdown(
    "<p style='color:#6b6b8a; font-size:1.05rem; margin-top:-10px;'>"
    "Enter songs you love — we'll find what plays next.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown("### 🎧 Your Input Songs")
st.markdown(
    "<p style='color:#6b6b8a; font-size:0.9rem;'>Enter 1–5 songs. "
    "Songs not in our dataset will be fetched from Spotify, so any era or genre works.</p>",
    unsafe_allow_html=True
)

with st.form("song_input_form"):
    entries = []
    for i in range(1, 6):
        col_name, col_year = st.columns([3, 1])
        with col_name:
            name = st.text_input(
                f"Song {i}",
                value=["Beat It", "Billie Jean", "Thriller", "", ""][i - 1],
                placeholder="Song title (leave blank to skip)",
                key=f"song_{i}"
            )
        with col_year:
            year = st.number_input(
                "Year",
                min_value=1900, max_value=2026,
                value=[1982, 1982, 1982, 1990, 1990][i - 1],
                key=f"year_{i}"
            )
        entries.append((name.strip(), int(year)))

    st.markdown("")
    submitted = st.form_submit_button("🚀 Get Recommendations", type="primary", use_container_width=True)

# ── Processing ─────────────────────────────────────────────────────────────────
if submitted:
    song_list = [{'name': n, 'year': y} for n, y in entries if n]

    if not song_list:
        st.warning("⚠️ Please enter at least one song title.")
        st.stop()

    with st.spinner("Analysing your taste profile and finding matches…"):
        recs, enriched_inputs = recommend_songs(song_list, data, n_recommendations)

    if recs is None or recs.empty:
        st.error("❌ Could not find recommendations. Try different song titles or check spelling.")
        st.stop()

    # ── Input Summary ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Input Songs Resolved")

    cols_input = st.columns(min(len(enriched_inputs), 5))
    for idx, info in enumerate(enriched_inputs):
        with cols_input[idx % len(cols_input)]:
            badge = (
                '<span class="badge-local">📦 dataset</span>'
                if info['source'] == 'dataset'
                else '<span class="badge-api">🌐 Spotify API</span>'
            )
            genre_pill = f'<span class="genre-pill">{info["genre"]}</span>' if info['genre'] != 'unknown' else ''
            st.markdown(f"""
            <div style="background:#131320; border:1px solid #2a2a45; border-radius:14px; padding:1rem 1.2rem;">
                <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; color:#e8e8f0; margin-bottom:4px;">{info['name']}</div>
                <div style="color:#6b6b8a; font-size:0.82rem; margin-bottom:8px;">{info['artists']} · {info['year']}</div>
                {badge} {genre_pill}
            </div>
            """, unsafe_allow_html=True)

    # ── Key Metrics ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Recommendation Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Songs Found",       len(recs))
    m2.metric("Avg Popularity",    f"{recs['popularity'].mean():.0f}/100")
    m3.metric("Year Range",        f"{int(recs['year'].min())}–{int(recs['year'].max())}")
    m4.metric("Unique Artists",    recs['artists'].nunique())
    m5.metric("Unique Genres",     recs['genre'].nunique())

    # ── Results Table ───────────────────────────────────────────────────────────
    st.markdown("### 🎵 Recommended Songs")

    display_df = recs.copy()
    display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.3f}")
    display_df['popularity']       = display_df['popularity'].apply(lambda x: f"{int(x)}/100")
    display_df.columns = ['#', 'Song', 'Year', 'Artists', 'Genre', 'Popularity', 'Similarity']

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Visualisations ──────────────────────────────────────────────────────────
    st.markdown("### 📈 Visual Analysis")

    tab1, tab2, tab3 = st.tabs(["🏆 Popularity", "📅 Timeline", "🎨 Audio Features"])

    # -- Tab 1: Popularity Bar Chart
    with tab1:
        fig_bar = px.bar(
            recs.sort_values('popularity', ascending=True),
            x='popularity',
            y='name',
            color='artists',
            orientation='h',
            title='Recommended Songs by Popularity Score',
            labels={'popularity': 'Popularity (0–100)', 'name': ''},
            color_discrete_sequence=px.colors.qualitative.Vivid,
            height=420,
        )
        fig_bar.update_layout(
            paper_bgcolor='#131320',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0', family='DM Sans'),
            title_font=dict(family='Syne', size=14, color='#e8e8f0'),
            xaxis=dict(gridcolor='#1e1e35', color='#6b6b8a'),
            yaxis=dict(gridcolor='#1e1e35', color='#e8e8f0'),
            legend=dict(bgcolor='#131320', bordercolor='#2a2a45', borderwidth=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # -- Tab 2: Timeline Scatter
    with tab2:
        fig_scatter = px.scatter(
            recs,
            x='year',
            y='popularity',
            color='artists',
            size='popularity',
            hover_data=['name', 'genre'],
            title='Recommendations Over Time',
            labels={'year': 'Release Year', 'popularity': 'Popularity'},
            color_discrete_sequence=px.colors.qualitative.Vivid,
            height=400,
        )
        fig_scatter.update_layout(
            paper_bgcolor='#131320',
            plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0', family='DM Sans'),
            title_font=dict(family='Syne', size=14, color='#e8e8f0'),
            xaxis=dict(gridcolor='#1e1e35', color='#6b6b8a'),
            yaxis=dict(gridcolor='#1e1e35', color='#6b6b8a'),
            legend=dict(bgcolor='#131320', bordercolor='#2a2a45', borderwidth=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -- Tab 3: Audio Feature Radar for top 3 recs
    with tab3:
        st.markdown("Audio feature profiles for the **top 3 recommendations**:")
        radar_cols = st.columns(3)
        for i, (_, row) in enumerate(recs.head(3).iterrows()):
            with radar_cols[i]:
                fig_radar = build_audio_radar(row, f"{row['name'][:22]}…" if len(row['name']) > 22 else row['name'])
                st.plotly_chart(fig_radar, use_container_width=True)

    # ── Deep Dive Expander ──────────────────────────────────────────────────────
    with st.expander("🔬 Deep Dive: Full Analysis"):
        st.markdown(f"""
        **Top Recommendation:** 🎵 **{recs.iloc[0]['name']}** by *{recs.iloc[0]['artists']}* ({int(recs.iloc[0]['year'])})

        **Most Popular Track:** 🔥 **{recs.loc[recs['popularity'].astype(float).idxmax(), 'name']}**  
        **Most Niche Track:** 💎 **{recs.loc[recs['popularity'].astype(float).idxmin(), 'name']}**  
        **Most Featured Artist:** 🎤 **{recs['artists'].mode()[0] if not recs['artists'].mode().empty else 'Various'}**

        ---
        **How to read these results:**
        - **Similarity score** (0–1): closeness to your taste profile — higher is more similar
        - Songs are sourced from a dataset of {len(data):,} tracks + live Spotify API fallback for anything not found locally
        - The algorithm matches on 15 audio features (energy, valence, tempo, acousticness, etc.)
        
        **Tips for better results:**
        - Mix songs from different years to get a broader taste profile
        - Use songs from the same genre to get tighter, more focused recommendations
        - Enter songs that aren't in the dataset to pull in fresh Spotify data
        """)

    st.markdown("---")
    st.caption("🎵 Spotify Music Recommender · Content-Based Filtering · Cosine Similarity · Powered by Spotify API")
