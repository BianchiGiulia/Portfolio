Multi Media Retrieval System Project
===============================

Overview
--------

This project centers on the development of an advanced, content-based music retrieval system. Utilizing an extensive dataset, it integrates a wide array of text-based features as its core components. The primary objective is to deepen the understanding and evaluation of these features, particularly in terms of their functionality and effectiveness. By doing so, the system aims to uncover and delineate the specific patterns that these features are best suited to retrieve. 

### Notebooks and Methods:

1.  **Text-Based Similarity and Retrieval** (`text_based_retrieval.ipynb`): This notebook demonstrates four distinct methods for text-based retrieval, including a Random Baseline and three systems based on cosine similarity using different embeddings: TF-IDF, Word2Vec, and BERT. Additionally, the BERT-based system incorporates the Manhattan distance as an alternative similarity measure. The methodology is consistent across these systems: initially, I generate a query vector using the selected embedding technique, followed by the retrieval of all songs, excluding the query song (noting that the 'id' column is excluded during this phase). Next, we calculate the similarity between the query and each song. The final step involves sorting the results by similarity, removing the top result (which is the query itself), and presenting the top N results along with their corresponding similarity scores.

2. **Audio based retrieval systems** (`audio_based_retrieval.ipynb`): To ensure compatibility with the existing codebase, I initially implemented a helper function (`find_song_info`). This function is designed to retrieve the artist and song information using their respective IDs, and vice-versa. A significant enhancement to this process is the incorporation of genre data. Now, for the top N songs identified by the system, we not only retrieve their IDs, artist names, and song titles, but also their respective genres. This results in a more comprehensive output: a list of the top N retrieved items, each item featuring its ID, artist name, song title, genre, and a measure of similarity. This addition enriches the understanding of each retrieved piece, allowing for broader evaluation measures (*see Evaluation and Results hereunder*).

Dataset
-------
This project leverages a specialized subset of the Music4All-Onion dataset. The comprehensive dataset is accessible at [Zenodo](https://zenodo.org/record/6609677#.Y0ff7XZBxD8), offering a rich collection of musical metadata and features.
In detail the file used in this project are:

**Text-based approach**: 
*  'id_information_mmsr.tsv': containing info about id,	artist,	song,	album_name.
*  'id_lyrics_tf-idf_mmsr.tsv': sparse matrix with tf-idf values.
*  'id_lyrics_bert_mmsr.tsv': BERT encoding of the lyrics.
*  'id_lyrics_word2vec_mmsr.tsv': Word2Vec encoding of the lyrics.

**Audio-based approach**: 

*  'id_information_mmsr.tsv':containing info about id,	artist,	song,	album_name.
*  'id_mfcc_stats_mmsr.tsv': Mel-Frequency Cepstral Coefficients with statistical descriptors.
*  'id_blf_logfluc_mmsr.tsv': Block-level features with logarithmic fluctuation.
*  'id_ivec256_mmsr.tsv': i-vectors of dimensionality 256.
*  'id_musicnn_mmsr.tsv': feature extracted via a CNN.
*  'id_genres_mmsr.tsv': info about songs genre.

Evaluation and Results
----------------------

While in the text-based approach the evaluation has been carried out by comparing the cosine similarity measure, the audio based approach allowed for broader and deeper insights. The following evaluation metrics have been implemented:

*  Average Precision@k and Recall@k, with k ranging [1,100].
*  Average nDCG@k, with k=10 and Sørensen–Dice coefficient.

  
And some more have been implemented specifically for the task:

*  Genre diversity@k: measures how evenly distributed are the genres over the top k retrieved tracks, with k=10.
*  Genre coverage@k: measures how many out of all genres present in the data (assigned to at least one track) are covered by retrieved tracks, with k=10.

All relative plots and tables can be found in the notebooks.



Contributions and Feedback
--------------------------

Your contributions and feedback are welcome! Feel free to fork this repository, experiment with the code, and provide suggestions or improvements.

* * *
