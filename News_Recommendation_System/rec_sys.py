
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#---------COLLABORATIVE FILTERING---------#

def pop_baseline(interaction_matrix, user_id, k=10, **kwargs):
   
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten() #calc pop
    interacted_indices = interaction_matrix[user_id].nonzero()[1] #already seen items
    non_interacted_items = np.setdiff1d(np.arange(interaction_matrix.shape[1]), interacted_indices, assume_unique=True) #filtered out
    non_interacted_popularity = item_popularity[non_interacted_items] #sort the remaining ones
    popular_non_interacted_items = non_interacted_items[np.argsort(non_interacted_popularity)[::-1]][:k]
    
    return popular_non_interacted_items #output: list of k items for 1 user


def item_based_rs(interaction_matrix, user_id, k=10, item_similarity=None, **kwargs):
    user_row = interaction_matrix[user_id]
    scores = user_row.dot(item_similarity).toarray().ravel() #precomputed in the eval functions
    ranking = scores.argsort()[::-1]
    interacted = interaction_matrix[user_id].nonzero()[1]
    ranking = [r for r in ranking if r not in interacted]
    return ranking[:k] 


def knn_item_based_rs(interaction_matrix, user_id, k=10, n=5, **kwargs):
    
    #fit model
    knn_model = NearestNeighbors(n_neighbors=n+1, algorithm='brute', metric='cosine')
    knn_model.fit(interaction_matrix.T)  
    
    interacted = interaction_matrix[user_id].nonzero()[1]
    item_distances, item_indices = knn_model.kneighbors(interaction_matrix.T, return_distance=True)
    
    
    scores = np.zeros(interaction_matrix.shape[1])#scores based on knn distance
    for idx in interacted:
        neighbors_indices = item_indices[idx][1:]
        neighbors_distances = item_distances[idx][1:]
        for neighbor_idx, distance in zip(neighbors_indices, neighbors_distances):
            scores[neighbor_idx] += 1 / (1 + distance)  # inverse distance weighting
    
    scores[interacted] = -np.inf #already seen
    recommended_indices = np.argsort(scores)[::-1][:k]
    
    return recommended_indices.tolist()