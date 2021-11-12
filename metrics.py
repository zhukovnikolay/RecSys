"""
Metrics

"""
import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[recommended_list <= k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)


    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant


def map_k(recommended_lists, bought_lists, k=5):
    
    sum_ = sum([ap_k(recommended_list, bought_list, k=k) for 
                recommended_list, bought_list in zip(recommended_lists, bought_lists)])
    
    result = sum_ / len(recommended_lists)
    
    return result


def dcg_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    
    if len(relevant_indexes) == 0:
        return 0

    sum_ = 0
    
    for i in relevant_indexes:
        
        discount = i + 1 if i <= 1 else np.log2(i + 1)
        sum_ += 1 / discount
    
    return sum_ / k


def ndcg_k(recommended_list, bought_list, k=5):
    
    ideal_dcg_k = dcg_k(bought_list, bought_list, k=k)
    result = dcg_k(recommended_list, bought_list, k=k) / ideal_dcg_k
    
    return result


def reciprocal_rank(recommended_list, bought_list):
    
    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    
    if len(relevant_indexes) == 0:
        return 0
    
    return 1 / (relevant_indexes[0] + 1)


def mrr(recommended_lists, bought_lists):
    
    sum_ = sum([reciprocal_rank(recommended_list, bought_list) for 
                recommended_list, bought_list in zip(recommended_lists, bought_lists)])
    
    result = sum_ / len(recommended_lists)
    
    return result