import numpy as np
import pandas as pd
import inspect
from scipy.sparse import find, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def dcg_at_k(r, k): #@TO-DO:RECHECK
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):#@TO-DO:RECHECK
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def evaluate(interaction_matrix, test_matrix, rec_functions, k=10, metric=None, **kwargs):  #default metric: cosine_similarity
    
    #----------PRECOMPUTE SIMILARITY MATRIX---------- 
    #for memory-based recommenders
    
    def adjusted_cosine_similarity(matrix):
        user_ratings_mean = matrix.mean(axis=1)
        mean_adj_matrix = matrix - csr_matrix(user_ratings_mean).transpose()
        return cosine_similarity(mean_adj_matrix, dense_output=False)
    
    #investigate more metrics here?
    
    def get_similarity(matrix, rec_func):
        matrix_to_use = matrix if rec_func.uses_similarity == 'item' else matrix.T
        if metric == 'adjusted cosine':
            return adjusted_cosine_similarity(matrix_to_use)
        else:
            return cosine_similarity(matrix_to_use, dense_output=False)
            
    
    #----------EVALUATE RECOMMENDERS----------
    results_list = []
    
    test_users, test_items_indices, _ = find(test_matrix) #row id, col id, data (only non-zero values)
    test_interactions = list(zip(test_users, test_items_indices)) #list of tuples of (user, item)

   
    user_to_test_items = {}                 #like {user: [item1, item2, ...]}
    for user, item in test_interactions:
        if user not in user_to_test_items:
            user_to_test_items[user] = set()
        user_to_test_items[user].add(item)

    
    for rec_func in rec_functions:
        precision_at_k_list = []
        recall_at_k_list = []
        ndcg_at_k_list = []
        f1_at_k_list = []
        #@TO-DO:
        #AUC
        #MMR
        
        similarity_matrix = get_similarity(interaction_matrix, rec_func)  
            
        #iterate & average across users
        for user_id, test_items in user_to_test_items.items():
            func_args = inspect.getfullargspec(rec_func).args
            if 'similarity_matrix' in func_args:
                recommended_items = rec_func(interaction_matrix, user_id, k=k, similarity_matrix=similarity_matrix, **kwargs)
            else:
                recommended_items = rec_func(interaction_matrix, user_id, k=k, **kwargs) #get recommendations
            hits_at_k = len(set(recommended_items) & test_items)  #int
            
            #actual evaluation
            precision_at_k = hits_at_k / k
            recall_at_k = hits_at_k / len(test_items) if test_items else 0 
            f1_at_k = 2*(precision_at_k*recall_at_k)/(precision_at_k+recall_at_k) if (precision_at_k+recall_at_k) else 0
            relevance_scores = [1 if item in test_items else 0 for item in recommended_items] #list: relevant if user actually clicked on it (list of 0 and 1)
            ndcg_score = ndcg_at_k(relevance_scores, k) #RECHECK
            #@TO-DO
            #AUC
            #MMR
            
            precision_at_k_list.append(precision_at_k)
            recall_at_k_list.append(recall_at_k)
            ndcg_at_k_list.append(ndcg_score)
            f1_at_k_list.append(f1_at_k)

        
        avg_precision_at_k = np.mean(precision_at_k_list)
        avg_recall_at_k = np.mean(recall_at_k_list)
        avg_ndcg_at_k = np.mean(ndcg_at_k_list)
        f1_at_k = np.mean(f1_at_k_list)
        
        results_list.append({
            'RecFunction': rec_func.__name__,
            'Precision@k': avg_precision_at_k,
            'Recall@k': avg_recall_at_k,
            'NDCG@k': avg_ndcg_at_k,
            'F1@k': f1_at_k,
        })
        print(rec_func.__name__, ": done")

    #display as df
    results_df = pd.DataFrame(results_list)
    
    return results_df
