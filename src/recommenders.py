import numpy as np
import pandas as pd

# Для работы с матрицами
from scipy.sparse import csr_matrix, coo_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from lightfm import LightFM

import os, sys
module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from src.metrics import precision_at_k, recall_at_k
from src.utils import prefilter_items

import warnings
warnings.filterwarnings("ignore")

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, item_features, user_features):
        self.prep_data = data   # self.prep_data = prefilter_items(data, item_features)
#         self.item_features = item_features
#         self.user_features = user_features
        print("user_item_matrix")
        self.user_item_matrix = self.prepare_matrix(self.prep_data)  # pd.DataFrame
        self.user_item_matrix_tfidf = self.prepare_matrix(self.prep_data)
        self.user_item_matrix_tfidf = tfidf_weight(self.user_item_matrix_tfidf.T).T
        self.user_item_matrix_bm25 = self.prepare_matrix(self.prep_data)
        self.user_item_matrix_bm25 = bm25_weight(self.user_item_matrix_bm25.T).T
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        print("user_feat")
        self.user_feat = pd.DataFrame(self.user_item_matrix.index)
        self.user_feat = self.user_feat.merge(user_features, on='user_id', how='left')
        self.user_feat.set_index('user_id', inplace=True)
        self.user_feat = self.user_feat.fillna("")
        self.user_feat_lightfm = pd.get_dummies(self.user_feat, columns=self.user_feat.columns.tolist())
        
        print("item_feat")
        self.item_feat = pd.DataFrame(self.user_item_matrix.columns)
        self.item_feat = self.item_feat.merge(item_features, on='item_id', how='left')
        self.item_feat.set_index('item_id', inplace=True)
        self.item_feat = self.item_feat.fillna("")
        self.item_feat_lightfm = pd.get_dummies(self.item_feat, columns=self.item_feat.columns.tolist())
        
        print("ALS model")
        self.model       = self.fit(self.user_item_matrix)
        print("ALS TF-IDF model")
        self.model_tfidf = self.fit(self.user_item_matrix_tfidf)
        print("ALS BM25 model")
        self.model_bm25  = self.fit(self.user_item_matrix_bm25)
        print("LightFM model")
        self.model_lightfm = self.fit_lightfm(self.user_item_matrix, self.user_feat_lightfm, self.item_feat_lightfm)
        print("Own")
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
    @staticmethod
    def prepare_matrix(data):

        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id',
                                          columns='item_id', 
                                          values='quantity', # Можно пробовать другие варианты
                                          aggfunc='count', 
                                          fill_value=0)

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=50, regularization=0.0001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)

        return model
    
    def fit_lightfm(self,
                    user_item_matrix, 
                    user_feat_lightfm, 
                    item_feat_lightfm, 
                    learning_rate=0.05, 
                    epochs=15, 
                    num_threads=4):
        """Обучает LightFM"""
        
        sparse_user_item = csr_matrix(user_item_matrix).tocsr()   # .astype(np.float)

        model = LightFM(no_components=5,
                        loss='bpr', # 'warp'
                        learning_rate=0.05, 
                        item_alpha=0.1, user_alpha=0.1, 
                        random_state=42)

        model.fit((sparse_user_item > 0) * 1,  # user-item matrix из 0 и 1
                  sample_weight=coo_matrix(user_item_matrix),
                  user_features=csr_matrix(user_feat_lightfm.values).tocsr(),
                  item_features=csr_matrix(item_feat_lightfm.values).tocsr(),
                  epochs=15, 
                  num_threads=4) 

        return model
    
    def get_als_recommendations(self, user, rec_num=5):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in 
                        self.model.recommend(userid=self.userid_to_id[user], 
                                             user_items=csr_matrix(self.user_item_matrix).tocsr(),   # на вход user-item matrix
                                             N=rec_num, 
                                             filter_already_liked_items=False, 
                                             filter_items=[self.itemid_to_id[999999]],  # !!! 
                                             recalculate_user=True)]
        return res

    def get_als_tfidf_recommendations(self, user, rec_num=5):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in 
                        self.model_tfidf.recommend(userid=self.userid_to_id[user], 
                                                 user_items=csr_matrix(self.user_item_matrix_tfidf).tocsr(), # tfidf
                                                 N=rec_num, 
                                                 filter_already_liked_items=False, 
                                                 filter_items=[self.itemid_to_id[999999]],  # !!! 
                                                 recalculate_user=True)]
        return res

    def get_als_bm25_recommendations(self, user, rec_num=5):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in 
                        self.model_bm25.recommend(userid=self.userid_to_id[user], 
                                                 user_items=csr_matrix(self.user_item_matrix_bm25).tocsr(), # bm25
                                                 N=rec_num, 
                                                 filter_already_liked_items=False, 
                                                 filter_items=[self.itemid_to_id[999999]],  # !!! 
                                                 recalculate_user=True)]
        return res
    
    def get_own_recommendations(self, user, rec_num=5):
        """Рекомендуем топ-N собственных товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in 
                    self.own_recommender.recommend(userid=self.userid_to_id[user], 
                                                   user_items=csr_matrix(self.user_item_matrix).tocsr(),   # на вход user-item matrix
                                                   N=rec_num, 
                                                   filter_already_liked_items=False, 
                                                   filter_items=[self.itemid_to_id[999999]],  # !!! 
                                                   recalculate_user=True)]
        return res
    
    def get_similar_items_recommendation(self, user, rec_num=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        popularity = self.prep_data.loc[self.prep_data["user_id"]==user, ['item_id', 'quantity']].\
                        groupby(['item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        top_item_list = popularity.loc[popularity["item_id"]!=999999, "item_id"][:rec_num].tolist()
        
        res = set()
        for item in top_item_list:
            res.add(self.id_to_itemid[self.model.similar_items(self.itemid_to_id[item], N=1)[0][0]])

#         assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return list(res)

    def get_similar_users_recommendation(self, user, rec_num=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        sim_users_list = [self.id_to_userid[usr] for usr, p in self.model.similar_users(self.userid_to_id[user], 
                                                                                        N=rec_num+1)[1:rec_num+1]]
        res = set()
        for user_in in sim_users_list:
            res.add(list(self.get_als_recommendations(user_in, rec_num=1))[0])
            
        i = 2
        while len(res) < rec_num:
            res.add(list(self.get_als_recommendations(sim_users_list[1], rec_num=i))[i-1])
            i += 1
        
#         assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return list(res)
