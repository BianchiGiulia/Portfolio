import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity
from math import log2




def find_top_n_similar_songs(feature_df, N, data_name): 
    print(f"Find TOP {N} recommendations for {data_name}...")
    start_time = time.time() 
    song_ids = feature_df['id']
    features = feature_df.drop(columns=['id']).values
    similarity_matrix = cosine_similarity(features)

    results = []

    for i in range(len(song_ids)):
        similarities = similarity_matrix[i]
        similarities[i] = -np.inf

        top_indices = np.argpartition(similarities, -N)[-N:]
        top_indices_sorted = top_indices[np.argsort(-similarities[top_indices])]

        for index in top_indices_sorted:
            results.append({
                'query_id': song_ids[i],
                'retrieved_id': song_ids[index],
                'similarity': similarities[index]})

    result_df = pd.DataFrame(results)
    print(f"DONE after {round(time.time()-start_time, 2)}s")
    print("-"*30)
    return result_df


def find_common_retrieved_songs(blf_data, w2v_data):
    merged_data = pd.merge(blf_data[['query_song', 'retrieved_song']], 
                           w2v_data[['query_song', 'retrieved_song']], 
                           on='query_song', 
                           suffixes=('_blf', '_w2v'))

    duplicates = merged_data[merged_data['retrieved_song_blf'] == merged_data['retrieved_song_w2v']]

    return duplicates[['query_song', 'retrieved_song_w2v', 'retrieved_song_blf']]

def remove_selected_stop_words_and_feat(text):
                    #manually selected stop words
    stop_words = {"feat.", "i", "you", "the", "and", "a", "an", "of"}

    words = text.split()
    filtered_text = ' '.join(word for word in words if word.lower() not in stop_words)
    return filtered_text

def increase_similarity_based_on_artist_and_song(blf_data, w2v_data, alpha=0.06, beta=1.0):
    
    def preprocess_and_adjust_similarity(df):
        
        df['query_artist_processed'] = df['query_artist'].apply(lambda x: remove_selected_stop_words_and_feat(str(x).lower()))
        df['retrieved_artist_processed'] = df['retrieved_artist'].apply(lambda x: remove_selected_stop_words_and_feat(str(x).lower()))
        df['query_song_processed'] = df['query_song'].apply(lambda x: remove_selected_stop_words_and_feat(str(x).lower()))
        df['retrieved_song_processed'] = df['retrieved_song'].apply(lambda x: remove_selected_stop_words_and_feat(str(x).lower()))

        
        song_match = df.apply(lambda x: any(word in x['retrieved_song_processed'] for word in x['query_song_processed'].split()), axis=1)
        artist_match = df.apply(lambda x: any(word in x['retrieved_artist_processed'] for word in x['query_artist_processed'].split()), axis=1)

       
        df.loc[song_match | artist_match, 'similarity'] += alpha

       
        df.drop(['query_artist_processed', 'retrieved_artist_processed', 'query_song_processed', 'retrieved_song_processed'], axis=1, inplace=True)

        return df
    
def concatenate_songs(blf_data, w2v_data, N, M):
    final_df = pd.DataFrame()

    blf_data['song_artist'] = blf_data['query_song'] + " - " + blf_data['query_artist']
    w2v_data['song_artist'] = w2v_data['query_song'] + " - " + w2v_data['query_artist']

    unique_song_artists = set(blf_data['song_artist'].unique()).union(set(w2v_data['song_artist'].unique()))

    for song_artist in unique_song_artists:
        blf_rows = blf_data[blf_data['song_artist'] == song_artist].head(N)
        w2v_rows = w2v_data[w2v_data['song_artist'] == song_artist].head(M)

        concatenated_rows = pd.concat([blf_rows, w2v_rows])
        final_df = pd.concat([final_df, concatenated_rows], ignore_index=True)

    final_df.drop('song_artist', axis=1, inplace=True)

    return final_df

def expand_dataset(df, full_dataset):
    query_info = pd.merge(df, full_dataset, left_on='query_id', right_on='id', how='left')
    query_info = query_info.rename(columns={'artist': 'query_artist', 'song': 'query_song', 'genre': 'query_genre'})

    retrieved_info = pd.merge(df, full_dataset, left_on='retrieved_id', right_on='id', how='left')
    retrieved_info = retrieved_info.rename(columns={'artist': 'retrieved_artist', 'song': 'retrieved_song', 'genre': 'retrieved_genre'})

    expanded_df = pd.DataFrame({
        'query_song': query_info['query_song'],
        'query_artist': query_info['query_artist'],
        'query_genre': query_info['query_genre'],
        'retrieved_song': retrieved_info['retrieved_song'],
        'retrieved_artist': retrieved_info['retrieved_artist'],
        'retrieved_genre': retrieved_info['retrieved_genre'],
        'similarity': df['similarity']})

    return expanded_df

#------------------------------------------------------------
# EVALUATION FUNCTIONS
#------------------------------------------------------------

def check_relevance(df): #only for precision
    df['query_genre'] = df['query_genre'].fillna('')
    df['retrieved_genre'] = df['retrieved_genre'].fillna('')
    df['query_genre_list'] = df['query_genre'].str.split(',')
    df['retrieved_genre_list'] = df['retrieved_genre'].str.split(',')
    df['relevant'] = df.apply(lambda x: any(genre in x['query_genre_list'] for genre in x['retrieved_genre_list']), axis=1)   
    return df

def calculate_precision_recall_at_k(df, k_max=100):
    precision_recall_at_k = {}

    def check_relevance_for_query(query_genre_list, retrieved_genre_list):
        return any(genre in query_genre_list for genre in retrieved_genre_list)

    total_relevant_per_query = {}
    for query_song, group in df.groupby('query_song'):
        query_genre_list = group.iloc[0]['query_genre_list']
        df['relevant'] = df.apply(lambda x: check_relevance_for_query(query_genre_list, x['retrieved_genre_list']), axis=1)
        total_relevant = df['relevant'].sum()
        total_relevant_per_query[query_song] = total_relevant

    for k in range(1, k_max + 1):
        precisions = []
        recalls = []

        for query_song, group in df.groupby('query_song'):
            top_k = group.sort_values(by='similarity', ascending=False).head(k)
            relevant_in_top_k = top_k['relevant'].sum()
            total_relevant = total_relevant_per_query[query_song]

            if total_relevant > 0:
                precision_at_k = relevant_in_top_k / k
                recall_at_k = relevant_in_top_k / total_relevant
            else:
                precision_at_k = 0
                recall_at_k = 0

            precisions.append(precision_at_k)
            recalls.append(recall_at_k)

        average_precision = np.mean(precisions)
        average_recall = np.mean(recalls)

        precision_recall_at_k[k] = (average_precision, average_recall)

    return precision_recall_at_k

def calculate_average_precision_at_k(data, k): 
    grouped_data = data.groupby('query_song')

    total_precision = 0
    num_queries = 0
    for query_name, group in grouped_data:
        if len(group) < k:
            raise ValueError(f"Not enough results for query {query_name} to calculate precision at {k}")
        precision_at_k = group.head(k)['relevant'].sum() / k
        total_precision += precision_at_k
        num_queries += 1
    average_precision = total_precision / num_queries if num_queries > 0 else 0
    return average_precision

def calculate_recall_at_k(df, k):
    def check_relevance_for_query(query_genre_list, retrieved_genre_list):
        return any(genre in query_genre_list for genre in retrieved_genre_list)

    total_recall = 0
    num_queries = 0

    for query_song, query_group in df.groupby('query_song'):
        query_genre_list = query_group.iloc[0]['query_genre_list']

        df['relevant'] = df.apply(lambda x: check_relevance_for_query(query_genre_list, x['retrieved_genre_list']), axis=1)

        top_k = query_group.sort_values(by='similarity', ascending=False).head(k)
        relevant_retrieved = top_k['relevant'].sum()
        total_relevant = df['relevant'].sum()

        if total_relevant > 0:
            recall = relevant_retrieved / total_relevant
            total_recall += recall
            num_queries += 1

    average_recall = total_recall / num_queries if num_queries > 0 else 0
    return average_recall


def sorenson_dice_coefficient(set1, set2):
    numerator = 2 * len(set1.intersection(set2))
    denominator = len(set1) + len(set2)
    return numerator / denominator if denominator != 0 else 0

def calculate_dcg(scores):
    return scores[0] + sum([scores[i] / log2(i + 2) for i in range(1, len(scores))])

def calculate_ndcg_at_10(berd_df):
    grouped = berd_df.groupby('query_song')
    total_ndcg = 0
    num_queries = 0

    for name, group in grouped:
        relevance_scores = [sorenson_dice_coefficient(set(group.iloc[i]['query_genre']), 
                                                     set(group.iloc[i]['retrieved_genre']))
                            for i in range(min(10, len(group)))]

        dcg_at_10 = calculate_dcg(relevance_scores)
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        idcg_at_10 = calculate_dcg(ideal_relevance_scores)

        ndcg_at_10 = dcg_at_10 / idcg_at_10 if idcg_at_10 != 0 else 0

        total_ndcg += ndcg_at_10
        num_queries += 1

    average_ndcg = total_ndcg / num_queries if num_queries > 0 else 0
    return average_ndcg


def genre_coverage_at_k(berd_df, k):
    all_genres = set().union(*berd_df['query_genre_list'], *berd_df['retrieved_genre_list'])

    grouped = berd_df.groupby('query_song')
    covered_genres = set()

    for name, group in grouped:
        top_k = group.head(k)
        for genres in top_k['retrieved_genre_list']:
            covered_genres.update(genres)

    genre_coverage = len(covered_genres) / len(all_genres) if all_genres else 0

    return genre_coverage

def calculate_entropy(distribution):
    return -sum(g * log2(g) for g in distribution if g > 0)

def genre_diversity_at_k(berd_df, k):
    all_genres = list(set().union(*berd_df['query_genre'], *berd_df['retrieved_genre']))
    genre_index = {genre: idx for idx, genre in enumerate(all_genres)}
    grouped = berd_df.groupby('query_song')

    total_entropy = 0
    num_queries = 0

    for name, group in grouped:
        genre_distribution = np.zeros(len(all_genres))

        for _, row in group.head(k).iterrows():
            for genre in row['retrieved_genre']:
                genre_distribution[genre_index[genre]] += 1 / len(row['retrieved_genre'])

        genre_distribution /= k
        entropy = calculate_entropy(genre_distribution)
        total_entropy += entropy
        num_queries += 1

    average_entropy = total_entropy / num_queries if num_queries > 0 else 0

    return average_entropy
