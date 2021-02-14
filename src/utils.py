import pandas as pd

def prefilter_items(data, item_features):
    # Оставим только 5000 самых популярных товаров
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
    #добавим, чтобы не потерять юзеров
    data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 999999
    
    # Уберем 50 самых популярных
    data.loc[data['item_id'].isin(top_5000[:50]), 'item_id'] = 999999
    
    # Уберем самые непопулряные
    popularity_pu = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity_pu.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity_pu['share_unique_users'] = popularity_pu['share_unique_users'] / data['user_id'].nunique()
    popularity_pu.sort_values('share_unique_users', ascending=False, inplace=True)
    unpop_list = popularity_pu[popularity_pu['share_unique_users'] < 0.01].item_id.tolist()
    data.loc[data['item_id'].isin(unpop_list), 'item_id'] = 999999
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    curr_week = data['week_no'].values.max()
    data.loc[data['week_no']<curr_week-53, 'item_id'] = 999999
    
    # Уберем не интересные для рекоммендаций категории (department)
    dep_list = ['DRUG GM', 'KIOSK-GAS', 'COUP/STR & MFG', 'PROD-WHS SALES',
                'RX', 'MEAT-WHSE', 'HBC', 'CNTRL/STORE SUP', 'POSTAL CENTER', 
                'PHOTO', 'VIDEO', 'PHARMACY SUPPLY']
    
    for dep in dep_list:
        drop_items_list = item_features.loc[item_features['department'] == dep, "item_id"].values
        data.loc[data['item_id'].isin(drop_items_list), 'item_id'] = 999999
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    mean_price = data.groupby('item_id')['quantity', "sales_value"].sum()
    mean_price["mean_price"] = mean_price["sales_value"] / mean_price["quantity"]
    cheap_list = mean_price.loc[mean_price["mean_price"]<0.9].index.tolist()
    data.loc[data['item_id'].isin(cheap_list), 'item_id'] = 999999
    
    # Уберем слишком дорогие товары
    luxury_list = mean_price.loc[mean_price["mean_price"]>1500].index.tolist()
    data.loc[data['item_id'].isin(luxury_list), 'item_id'] = 999999
    
    # ...
    
    return data


def postfilter_items():
    pass
