import streamlit as st  
from PIL import Image
import os

# Get the current directory of this file
current_dir = os.path.dirname(__file__)

# Helper function to load images safely
def load_image(filename):
    path = os.path.join(current_dir, filename)
    return Image.open(path)

st.title("📊 Data Explorer")
st.subheader("Explore the Spotify Music Dataset and Visualize the distribution of tracks ")
st.subheader("Visualize the distribution of tracks across different decades ")

# --- First Plot ---
image = load_image("output.png")
st.image(image, use_column_width=True)
st.write("""The output of the code is a count plot (a bar chart) that shows the distribution of tracks across different decades in your dataset.

Why the counts decrease after some decades

There are a few possible reasons:

1. Dataset bias (most common)
The dataset (e.g., Spotify dataset) might contain more recent music and fewer old tracks. Streaming platforms usually prioritize modern music catalogs, so counts in early decades (1950s, 1960s) are often underrepresented.

2. Music industry changes
* Earlier decades (e.g., 1960s–1980s) had fewer global releases, physical limitations (vinyl, cassettes), and lower documentation.
* In the 2000s–2010s, the digital era allowed a huge surge in music production, so counts peak.
* In the 2020s (if included), the data may look smaller simply because the decade isn’t finished yet.

3. Data truncation
* Some datasets stop at a certain year (e.g., up to 2019), so the last decade might look incomplete.

4. Genre and popularity shifts
* Certain decades had more popular music production (e.g., 1980s pop, 1990s rock), while others had less mainstream output.
""")

# --- Trends of Features ---
st.subheader("Plot the trends of various sound features over the years")
image = load_image("newplot.png")
st.image(image, use_column_width=True)

st.subheader("Why do features increase or decrease?")
st.write("""Each feature represents a musical quality, and the ups/downs reflect changes in music styles, technology, and culture:

1. Acousticness
* High in earlier decades (folk, jazz, rock with acoustic instruments).
* Decreases with the rise of electronic & synthesized music in 1980s–2000s.
* May rise again in 2010s due to indie/folk revival

2. Danceability
* Lower in old ballads/rock.
* Rises in disco era (1970s), EDM/hip-hop (1990s–2010s)
* Fluctuates depending on popularity of dance-oriented genres

3. Energy
* Rises with rock, punk, EDM (1980s–2010s).
* May drop in 2010s–2020s when lo-fi, mellow hip-hop, and chill pop became trendy.

4. Instrumentalness
* High when instrumental genres (jazz, classical crossovers, progressive rock) were more popular.
* Drops in modern decades since vocals dominate mainstream music.

5. Liveness
* Spikes in decades with live-recording styles (1970s rock concerts, MTV Unplugged in 1990s).
* Lower in studio-polished music (2000s–2010s).

6. Valence (positiveness/happiness)
* High in disco/pop eras (1970s–1980s).
* Drops during grunge/alternative rock era (1990s)
* Mixed trends in 2010s with both upbeat EDM and darker hip-hop/lo-fi.
""")

# --- Loudness Trend ---
st.subheader("Plot the trend of loudness over decades using a line plot")
image = load_image("line.png")
st.image(image, caption="Track Distribution", use_column_width=True)
st.write("""Loudness in music has increased over the decades due to several factors:
* Advances in recording/mastering tech
* Industry competition
* Rise of genres that favor compressed, punchy, loud mixes
""")

# --- Genres Analysis ---
st.subheader("Identify the top 10 genres based on popularity and plot the trends of various sound features")
image = load_image("genres.png")
st.image(image, caption="Track Distribution", use_column_width=True)
st.subheader("Typical insights from such a chart")
st.write("""1. Valence (positiveness/happiness) – Higher in upbeat genres like pop, dance, disco. Lower in moodier genres like hip-hop, metal, alternative rock.
2. Energy – High in EDM, metal, hip-hop, rock (driving beats, loud production). Lower in folk, acoustic, indie.
3. Danceability – Peaks in dance, pop, hip-hop, reggaeton. Lower in rock, classical, jazz fusion.
4. Acousticness – High in folk, indie. Very low in EDM, hip-hop, pop.
""")

# --- Genre Word Cloud ---
st.subheader("Generate a word cloud of the genres present in the dataset")
image = load_image("genre.png")
st.image(image, caption="Track Distribution", use_column_width=True)

# --- Artist Word Cloud ---
st.subheader("Generate a word cloud of the artist present in the dataset")
image = load_image("artist.png")
st.image(image, caption="Track Distribution", use_column_width=True)

# --- t-SNE Visualization ---
st.subheader("Visualize the distribution of tracks using t-SNE")
image = load_image("tsne.png")
st.image(image, caption="Track Distribution", use_column_width=True)

# --- PCA Visualization ---
st.subheader("Visualize the distribution of tracks using PCA")
image = load_image("pca.png")
st.image(image, caption="Track Distribution", use_column_width=True)
