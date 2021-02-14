from implicit.nearest_neighbours import ItemItemRecommender 
from scipy.sparse import csr_matrix


def prefilter_items(data, item_features, take_n_popular=5000):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() 
    popularity['share_unique_users'] = popularity['user_id'] / data['user_id'].nunique()
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    
    # Уберем не интересные для рекоммендаций категории (department)
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    
    # Уберем слишком дорогие товарыs
    
    # ...
    return data
    
def postfilter_items(user_id, recommednations):
    pass

def get_rec(model, x, itemid_to_id, id_to_itemid):
    
    recs = model.similar_items(itemid_to_id[x], N=2)
    top_rec = recs[1][0]
    return id_to_itemid[top_rec]

def get_recommendations(user, model, sparse_user_item, itemid_to_id, id_to_itemid, userid_to_id, N=5):
    """Рекомендуем топ-N товаров"""
    
    res = [id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=userid_to_id[user], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=None,  # !!! 
                                    recalculate_user=True)]
    return res


def get_similar_items_recommendation(data, user, model, itemid_to_id, id_to_itemid, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    
    popularity= data[data['user_id']==user]
    popularity = popularity.groupby(['item_id'])['quantity'].count().reset_index()
    popularity.sort_values('quantity', ascending=False, inplace=True)
    popularity = popularity.head(N)   
    popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: get_rec(model, x, itemid_to_id, id_to_itemid))
    res = popularity['similar_recommendation'].to_list()
    
    return res

def get_similar_users_recommendation(user, model, user_item_matrix, itemid_to_id, id_to_itemid, userid_to_id, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    similar_users = model.similar_users(userid_to_id[user], N+1)
    
    own = ItemItemRecommender(K=1, num_threads=4) # K - кол-во билжайших соседей
    own.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
    
    res = []
    #первый пользователь - это он сам    
    for i in range(len(similar_users)):
        if i==0:
            continue
        sparse_user_item = csr_matrix(user_item_matrix).tocsr()
        rec = get_recommendations(i, own, sparse_user_item, itemid_to_id, id_to_itemid, userid_to_id, N=1)
        res.append(rec[0])
    
    return res
