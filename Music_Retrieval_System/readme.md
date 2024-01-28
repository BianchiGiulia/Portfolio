Multi Media Retrieval System Project
===============================

Overview
--------

This project centers on the development of an advanced, content-based music retrieval system. The dataset consists of ∼10,100 songs; 3 text feature embedding (tf-idf, BERT and Word2Vec), 4 audio embedding (MFCC, BLF, i-VEC and DNN) and 1 video feature (VGG19). We retrieve 100 items for each query, for a sum of ∼1,000,100 retrieved items for each of the 8 vector-space model based retrieval systems. We also implement a random baseline and two fusion model, for a total of 11 retrieval systems. The performance of music retrieval systems is evaluated using 5 different metrics: Precision@10, Recall@10, nDCG@10, Genre Coverage@10, and Genre Diversity@10. 

### Notebooks and Methods:

1.  **Music RS** (`MMRS.ipynb`): This notebook shows the output of 3 selected RS: VGG19, early and late fusion. It also contains the evaluation table and plot.

2. **Utility function** (`utils2.py`): Contain all the retrieval, processing, evaluation functions. For more details see *Experimental Setup*

Dataset
-------
This project leverages a specialized subset of the Music4All-Onion dataset. The comprehensive dataset is accessible at [Zenodo](https://zenodo.org/record/6609677#.Y0ff7XZBxD8), offering a rich collection of musical metadata and features.
In detail the file used in this project are:

*  'id_information_mmsr.tsv': containing info about id,	artist,	song,	album_name.
*   'id_genres_mmsr.tsv': info about songs genre.
   
**Text-based approach**: 

*  'id_lyrics_tf-idf_mmsr.tsv': sparse matrix with tf-idf values.
*  'id_lyrics_bert_mmsr.tsv': BERT encoding of the lyrics.
*  'id_lyrics_word2vec_mmsr.tsv': Word2Vec encoding of the lyrics.

**Audio-based approach**: 

*  'id_mfcc_stats_mmsr.tsv': Mel-Frequency Cepstral Coefficients with statistical descriptors.
*  'id_blf_logfluc_mmsr.tsv': Block-level features with logarithmic fluctuation.
*  'id_ivec256_mmsr.tsv': i-vectors of dimensionality 256.
*  'id_musicnn_mmsr.tsv': feature extracted via a CNN.
  
**Video-based approach**: 

*  'id_vgg19_mmsr.tsv': feature extracted via a 19 layers deep CNN.

Methodology
----------------------
*  **Early fusion tecnhique**: <br>
  The early fusion model is implemented with Word2Vec and MFCC features, denoted as Word2Vec + Mel Frequency Cepstral Coefficient. The decision to combine these specific features was based on their superior performance relative to other feature combinations. Word2Vec stood out as the top performer within the text feature set, while Mel Frequency Cepstral Coefficients yielded the best results through a comprehensive experimental approach, measuring the averaged similarity of the
first retrieved item for each query.
In the initial phase of the early fusion process, data normalization was prioritized. The rationale be- hind employing normalization, specifically the Euclidean norm (L2), stemmed from the necessity for a uniform the scale of the embeddings. Furthermore, the L2 norm ensured that the scaled magnitude would not alter the direction of vectors, aligning well with the requirements of cosine similarity. Following normalization, various dimensionality reduction methods were explored, including Princi- pal Component Analysis (PCA) and a combination of Post Processing Algorithm(PPA, from [this paper](https://aclanthology.org/W19-4328/))  + PCA. Dimensionality reduction methods aim to reduce the number of features in a dataset while preserving its essential information. However, attempts at dimensionality reduction resulted in a substantial information loss, exceeding 20% (based on the retained variance), therefore we decided to not pursue the research further and mantain the actual dimension of both features.
The justification for utilizing cosine similarity is because it is scale invariant, therefore useful to assess the impact of each feature on the results. Additionally it is effective in processing continuous feature vectors and is less impacted by the "curse of dimensionality". Lastly, it is straightforward, efficient and often used in the field of information retrieval.
*  **Late fusion tecnhique**: <br> For the late fusion model, we opted again for text and audio feature combination, concatenating the top N results from Word2Vec (R1) and top N from BLF (R2) retrieval systems, as the latter are effective in capturing spectral, harmonic, rhythmic, and tonal music characteristics. We could not use rank based nor score based algorithm, because each pair of sets of results had the same item among the retrieved songs only 1.8% of the cases. Therefore we implemented an algorithm which favors some item based on a few assumptions. Firstly, we assume that audio feature are slightly more important to retrieve similar songs than text based ones, therefore the final concatenated results are 60% from R2 and 40% from R1. We also assume that song retrieved from both methods for the same query have the highest likelihood to be similar to the query despite their rank (as they have been deemed similar notwithstanding using different feature set from different categories), therefore for such item we improve the similarity score of value β. Lastly, we assume that song with same words in the title are semantically linked (textual embedding is created from lyrics only), therefore also the similarity value of this item is incremented by α (see Section 5 for further details). All those addition are computed singularly for R1 and R2 (except the common retrieved songs) before concatenation and re-ordering.

Experimental Setup
----------------------
The cosine similarity used in the main retrieval function, the L2 normalization and PCA functions used in the early fusion model are both from the sklearn library, all the evaluation function (see Section 6) and the PPA algorithm are implemented from scratch, in Python.
The PCA was tested for various feature combination with n_components = d/2 , where d is the dimensionality of the feature embedding. In the late fusion model the parameter α is set to 0.06 when N=100, and 0.003 for N=10, both number are calculated as the average (over all queries) difference between the first and the Nth retrieved item in the BFL result (R2) (we choose this last one because as explained in Section 4, it is the model that ’participates’ the most in the final result), while β = 1.0 always. It must be noted that the optimization of this parameter has not been researched due to time constraints. It is also worth mentioning, that giving such a big value to β leads to similarity value to not be bounded anymore between [-1;1], which is not an issue within the scope of this project, as none of the evaluation function hereunder discussed is based on the similarity values computed. Lastly, in the comparison between title of songs, we excluded some manually selected stop-words: some articles, pronouns and preposition, plus the word "feat.".

Evaluation and Results
----------------------
Hereunder I report the table with the results of all 11 RS on 5 different metrics:
<img width="661" alt="Screenshot 2024-01-28 at 12 38 51" src="https://github.com/Giuliasdfghjk/Portfolio/assets/80102658/e446f967-051e-4436-b76d-16aa7db395a7">


Contributions and Feedback
--------------------------

Your contributions and feedback are welcome! Feel free to fork this repository, experiment with the code, and provide suggestions or improvements.

* * *
