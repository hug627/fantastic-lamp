import streamlit as st  
from PIL import Image
import os

# Get the current directory of this file
current_dir = os.path.dirname(__file__)

# Helper function to load images safely
def load_image(filename):
    path = os.path.join(current_dir, filename)
    if os.path.exists(path):
        return Image.open(path)
    else:
        st.error(f"Image not found: {filename}")
        return None

st.title("📊 Data Explorer")
st.subheader("Explore the Spotify Music Dataset and Visualize the distribution of tracks")
st.subheader("Visualize the distribution of tracks across different decades")

# --- First Plot ---
image = load_image("output.png")
if image:
    st.image(image, use_container_width=True)
    st.write("""The output of the code is a count plot (a bar chart) that shows the distribution of tracks across different decades in your dataset.

**Why the counts decrease after some decades**

There are a few possible reasons:

1. **Dataset bias (most common)**
   - The dataset (e.g., Spotify dataset) might contain more recent music and fewer old tracks. Streaming platforms usually prioritize modern music catalogs, so counts in early decades (1950s, 1960s) are often underrepresented.

2. **Music industry changes**
   - Earlier decades (e.g., 1960s–1980s) had fewer global releases, physical limitations (vinyl, cassettes), and lower documentation.
   - In the 2000s–2010s, the digital era allowed a huge surge in music production, so counts peak.
   - In the 2020s (if included), the data may look smaller simply because the decade isn't finished yet.

3. **Data truncation**
   - Some datasets stop at a certain year (e.g., up to 2019), so the last decade might look incomplete.

4. **Genre and popularity shifts**
   - Certain decades had more popular music production (e.g., 1980s pop, 1990s rock), while others had less mainstream output.
""")

# --- Trends of Features ---
st.subheader("Plot the trends of various sound features over the years")
image = load_image("newplot.png")
if image:
    st.image(image, use_container_width=True)

    st.subheader("Why do features increase or decrease?")
    st.write("""Each feature represents a musical quality, and the ups/downs reflect changes in music styles, technology, and culture:

1. **Acousticness**
   - High in earlier decades (folk, jazz, rock with acoustic instruments).
   - Decreases with the rise of electronic & synthesized music in 1980s–2000s.
   - May rise again in 2010s due to indie/folk revival

2. **Danceability**
   - Lower in old ballads/rock.
   - Rises in disco era (1970s), EDM/hip-hop (1990s–2010s)
   - Fluctuates depending on popularity of dance-oriented genres

3. **Energy**
   - Rises with rock, punk, EDM (1980s–2010s).
   - May drop in 2010s–2020s when lo-fi, mellow hip-hop, and chill pop became trendy.

4. **Instrumentalness**
   - High when instrumental genres (jazz, classical crossovers, progressive rock) were more popular.
   - Drops in modern decades since vocals dominate mainstream music.

5. **Liveness**
   - Spikes in decades with live-recording styles (1970s rock concerts, MTV Unplugged in 1990s).
   - Lower in studio-polished music (2000s–2010s).

6. **Valence (positiveness/happiness)**
   - High in disco/pop eras (1970s–1980s).
   - Drops during grunge/alternative rock era (1990s)
   - Mixed trends in 2010s with both upbeat EDM and darker hip-hop/lo-fi.
""")

# --- Loudness Trend ---
st.subheader("Plot the trend of loudness over decades using a line plot")
image = load_image("line.png")
if image:
    st.image(image, caption="Track Distribution", use_container_width=True)
    st.write("""**Loudness in music has increased over the decades** due to several factors:

The chart shows a steady increase in loudness over decades because of:
- Advances in recording/mastering tech
- Industry competition (the "loudness war")
- Rise of genres that favor compressed, punchy, loud mixes""")

# --- Genres Analysis ---
st.subheader("Identify the top 10 genres based on popularity and plot the trends of various sound features")
image = load_image("genres.png")
if image:
    st.image(image, caption="Track Distribution", use_container_width=True)
    st.subheader("Typical insights from such a chart")
    st.write("""1. **Valence (positiveness/happiness)**
   - Higher in upbeat genres like pop, dance, disco.
   - Lower in moodier genres like hip-hop, metal, alternative rock.

2. **Energy**
   - High in EDM, metal, hip-hop, rock (driving beats, loud production).
   - Lower in folk, acoustic, indie.

3. **Danceability**
   - Peaks in dance, pop, hip-hop, reggaeton
   - Lower in genres like rock, classical, jazz fusion.

4. **Acousticness**
   - High in folk, indie.
   - Very low in EDM, hip-hop, pop (dominated by electronic production).

**Summary of the chart**
1. The chart shows feature profiles of the 10 most popular genres.
2. Genres that are party-oriented (EDM, dance, pop, reggaeton) tend to score high in energy & danceability, low in acousticness.
3. Genres that are acoustic/folk-based show the opposite: high acousticness, lower energy & danceability.
4. Valence (positiveness) fluctuates: some genres are naturally happier (pop/dance), while others are darker (metal/hip-hop).""")

# --- Genre Word Cloud ---
st.subheader("Generate a word cloud of the genres present in the dataset")
image = load_image("genre.png")
if image:
    st.image(image, caption="Track Distribution", use_container_width=True)
    st.write("A word cloud is a visual representation of text data where the size of each word indicates its frequency or importance. In this case, the word cloud represents the genres present in the Spotify music dataset.")
    st.write("""**Summary**

The word cloud gives a visual snapshot of which genres dominate your dataset.

1. **Larger words** = more tracks in that genre (higher frequency).
2. **Smaller words** = niche genres that appear less often.

**Typical Spotify dataset word cloud:**
1. Pop, Rock, indie → very large (dominant mainstream genres).
2. Hip-hop, Metal, Folk → medium-large.
3. Irish, Danish, Reggae → smaller (less frequent).""")

# --- Artist Word Cloud ---
st.subheader("Generate a word cloud of the artists present in the dataset")
image = load_image("artist.png")
if image:
    st.image(image, caption="Track Distribution", use_container_width=True)
    st.write("""**Summary of the output**

The word cloud shows which artists dominate your dataset.

1. **Large artist names** = artists with many tracks in the dataset.
2. **Smaller artist names** = less frequent artists (appear in fewer tracks).

**Typical Spotify dataset word cloud:**
1. Orchestra, William, Johnny → very large (prolific artists with many tracks).
2. Trio, Brown, King → medium
3. Adam, Roy, Tom → smaller (less frequent).""")

# --- t-SNE Visualization ---
st.subheader("Visualize the distribution of tracks using t-SNE")
image = load_image("tsne.png")
if image:
    st.image(image, caption="Track Distribution", use_container_width=True)
    st.write("""**t-SNE** (t-distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that helps visualize high-dimensional data in 2D or 3D space. In this case, it's used to visualize the distribution of tracks based on their audio features.""")
    st.write("""**Summary of the output**
1. Each point = a track in your dataset.
2. Points close together = similar audio features (e.g., tempo, energy, danceability).
3. Clusters = groups of similar tracks (e.g., same genre, mood).
4. Outliers = unique tracks that don't fit common patterns.
""")
    st.write("""**Typical insights from such a chart**
1. Clusters of points indicate groups of similar tracks based on audio features.
2. Different colors (if used) can represent genres, decades, or popularity levels.
3. Dense clusters suggest popular styles/genres with many similar tracks.
4. Sparse areas or outliers may represent niche or unique tracks.
""")

# --- PCA Visualization ---
st.subheader("Visualize the distribution of tracks using PCA")
image = load_image("pca.png")
if image:
    st.image(image, caption="Track Distribution", use_container_width=True)
    st.write("""**PCA** (Principal Component Analysis) is another dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. In this case, it's used to visualize the distribution of tracks based on their audio features.""")
    
    st.write("""**What the code does:**

1. **Prepares data**
   - Selects only numeric song features (e.g., energy, danceability, loudness, tempo).
   - Drops missing values so the analysis runs smoothly.

2. **Standardizes the features**
   - Uses StandardScaler so all features are on the same scale (mean = 0, std = 1).
   - Prevents features like loudness (dB) from overpowering smaller-scale features like valence (0–1).

3. **Clusters songs with KMeans**
   - Groups the songs into 5 clusters (n_clusters=5), where each cluster contains songs with similar musical attributes.
   - Each song gets a cluster label (0–4).

4. **Applies PCA (dimensionality reduction)**
   - Reduces all numeric song features to 2 principal components (PC1, PC2).
   - These PCs capture most of the variance in the data, making it easier to visualize.

5. **Creates a visualization dataset**
   - Builds a DataFrame with PC1, PC2 (reduced coordinates), Cluster (assigned label), Title (song name), and Artist (artist name).

6. **Plots with Plotly**
   - An interactive scatter plot is generated where X-axis = PC1, Y-axis = PC2, colors = different clusters (0–4), and hover tooltip shows the song's title and artist.

**What the output looks like:**
- A 2D scatter plot where each dot = one song.
- Songs are grouped into 5 distinct colored clusters.
- Songs in the same cluster are close together in the plot, meaning they share similar musical features.
- Hovering over a dot shows which song and artist it is, letting you explore individual tracks.

**Summary:**
The code produces an interactive PCA scatter plot of songs, where songs are grouped into 5 clusters based on their audio features. Each cluster likely represents a different style of music (e.g., high-energy dance tracks vs. mellow acoustic songs). The plot allows you to explore these groupings visually and inspect specific songs by hovering.""")
