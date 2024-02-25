
import numpy as np
from scipy.sparse.linalg import svds


#---------COLLABORATIVE FILTERING---------#

def pop_baseline(interaction_matrix, user_id, k=10, **kwargs):
   
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten() #calc pop
    interacted_indices = interaction_matrix[user_id].nonzero()[1] #already seen items
    non_interacted_items = np.setdiff1d(np.arange(interaction_matrix.shape[1]), interacted_indices, assume_unique=True) #filtered out
    non_interacted_popularity = item_popularity[non_interacted_items] #sort the remaining ones
    popular_non_interacted_items = non_interacted_items[np.argsort(non_interacted_popularity)[::-1]][:k]
    
    return popular_non_interacted_items #output: list of k items for 1 user
pop_baseline.uses_similarity = 'none'

# memory  based    

def itembased_rs(interaction_matrix, user_id, k, similarity_matrix, **kwargs):
    user_row = interaction_matrix[user_id] #1 x n_items
    scores = user_row.dot(similarity_matrix).toarray().ravel() #precomputed in the eval function
    ranking = np.argsort(scores)[::-1]
    interacted = interaction_matrix[user_id].nonzero()[1]
    recommended_indices = [i for i in ranking if i not in interacted][:k]
    return recommended_indices
itembased_rs.uses_similarity = 'item'

def userbased_rs(interaction_matrix, user_id, k, similarity_matrix, **kwargs):
    scores = similarity_matrix[user_id].dot(interaction_matrix).toarray().ravel() #precomputed transpose
    ranking = np.argsort(scores)[::-1]
    interacted = interaction_matrix[user_id].nonzero()[1]
    recommended_indices = [i for i in ranking if i not in interacted][:k]
    return recommended_indices
userbased_rs.uses_similarity = 'user'


# model based

def svd_decompose_sparse(interaction_matrix, f=None):
    f = min((interaction_matrix.shape))-1 if f is None else f #n of factors= min(n_users, n_items)-1
    U, s, Vt = svds(interaction_matrix, k=f)
    U, s, Vt = U[:, ::-1], s[::-1], Vt[::-1, :] #rearrange
    U = U @ np.diag(np.sqrt(s)) #scaling
    V = (np.diag(np.sqrt(s)) @ Vt).T
    return U, V

def svd_recommend_to_list(interaction_matrix, user_id, k, f=None):
    U, V = svd_decompose_sparse(interaction_matrix, f)
    scores = U @ V.T
    u_scores = scores[user_id]
    interacted = interaction_matrix[user_id].nonzero()[1]
    u_scores[interacted] = -np.inf #exclude interacted items
    recs = np.argsort(u_scores)[:k][::-1]
    return recs
svd_recommend_to_list.uses_similarity = 'none'

#TO-DO: BAYESIAN PERSONALIZED RANKING
