Multi Media Retrieval System Project
===============================

Overview
--------

This project centers on the development of an advanced, content-based music retrieval system. Utilizing an extensive dataset, it integrates a wide array of text-based features as its core components. The primary objective is to deepen the understanding and evaluation of these features, particularly in terms of their functionality and effectiveness. By doing so, the system aims to uncover and delineate the specific patterns that these features are best suited to retrieve. 

### Notebooks and Methods:

1.  **Text-Based Similarity and Retrieval** (`text_based_retrieval.ipynb`): This notebook demonstrates four distinct methods for text-based retrieval, including a Random Baseline and three systems based on cosine similarity using different embeddings: TF-IDF, Word2Vec, and BERT. Additionally, the BERT-based system incorporates the Manhattan distance as an alternative similarity measure. The methodology is consistent across these systems: initially, I generate a query vector using the selected embedding technique, followed by the retrieval of all songs, excluding the query song (noting that the 'id' column is excluded during this phase). Next, we calculate the similarity between the query and each song. The final step involves sorting the results by similarity, removing the top result (which is the query itself), and presenting the top N results along with their corresponding similarity scores.

2. **Audio based retrieval systems** (`audio_based_retrieval.ipynb`): To ensure compatibility with the existing codebase, I initially implemented a helper function (`find_song_info`). This function is designed to retrieve the artist and song information using their respective IDs, and vice-versa. A significant enhancement to this process is the incorporation of genre data. Now, for the top N songs identified by the system, we not only retrieve their IDs, artist names, and song titles, but also their respective genres. This results in a more comprehensive output: a list of the top N retrieved items, each item featuring its ID, artist name, song title, genre, and a measure of similarity. This addition enriches the understanding of each retrieved piece, allowing for broader evaluation (*see results*).

Dataset
-------
