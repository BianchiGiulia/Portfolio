import streamlit as st
import pandas as pd

# Load data
path_to_csv = "early_fusion_top_10_similar_songs.csv"
path_to_url = "id_url_mmsr.tsv"
path_to_info = "id_information_mmsr.tsv"

fusion_recommendations = pd.read_csv(path_to_csv)
song_urls = pd.read_csv(path_to_url, sep="\t")
song_infos = pd.read_csv(path_to_info, sep="\t")


# Fancy header
st.title("🎵 🎵 🎵 Music Recommendation")
unique_artists = song_infos['artist'].unique()
selected_artist = st.selectbox("Select Artist", unique_artists)
filtered_songs = song_infos[song_infos['artist'] == selected_artist]
selected_song = st.selectbox("Select Song", filtered_songs['song'].unique())
selected_id = filtered_songs[filtered_songs['song'] == selected_song]['id'].values[0]
recommendations = fusion_recommendations[fusion_recommendations['query_id'] == selected_id]

merged_recommendations = pd.merge(recommendations, song_infos, left_on='retrieved_id', right_on='id', how='left')
merged_recommendations2 = pd.merge(merged_recommendations, song_urls, left_on='retrieved_id', right_on='id', how='left')

# User selection section in a table-like style
st.write(f"**Selections:**")
st.write(f"| **Artist**  ➡️  {selected_artist} |", f"| **Song**  ➡️  {selected_song} |")


# Display table-like structure using st.markdown
table_content = "| Artist | Song | Album | Similarity |\n| --- | --- | --- | --- |\n"

for index, row in merged_recommendations2.iterrows():
    table_content += f"| {row['artist']} | [{row['song']}]({row['url']}) | {row['album_name']} | {row['similarity']} |\n"

st.markdown(table_content)
