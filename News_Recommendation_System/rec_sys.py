
import numpy as np
from scipy.sparse import csr_matrix

#---------COLLABORATIVE FILTERING---------#

def pop_baseline(interaction_matrix: csr_matrix, user_id: int, k=10):
   
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten() #calc pop
    interacted_indices = interaction_matrix[user_id].nonzero()[1] #already seen items
    non_interacted_items = np.setdiff1d(np.arange(interaction_matrix.shape[1]), interacted_indices, assume_unique=True) #filtered out
    non_interacted_popularity = item_popularity[non_interacted_items] #sort the remaining ones
    popular_non_interacted_items = non_interacted_items[np.argsort(non_interacted_popularity)[::-1]][:k]
    
    return popular_non_interacted_items #output: list of k items for 1 user