import pandas as pd
import numpy as np

def prefilter_items(data, item_features, take_n_popular=5000):
    
    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    sold_last_52_weeks = data[data['week_no'] > data['week_no'].max()-52].item_id.unique().tolist()
    data.loc[~data['item_id'].isin(sold_last_52_weeks), 'item_id'] = 999999
    
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        not_allowed_cat = ['SPIRITS']
        additional_filter = item_features['department'].isin(not_allowed_cat)
        # фильтруем категории с количеством товаров < 25
        mask = (item_features.groupby('department')['department'].transform(len) > 25) & (~additional_filter)
        allowed_items_id = item_features[mask].item_id.tolist()
        data.loc[~data['item_id'].isin(allowed_items_id), 'item_id'] = 999999

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data_copy = data.copy()
    cheap_criterion = 2
    data_copy['price'] = data_copy['sales_value'] / (np.maximum(data_copy['quantity'], 1))
    cheap_items_id = data_copy[data_copy['price'] < cheap_criterion].item_id.tolist()
    data.loc[data['item_id'].isin(cheap_items_id), 'item_id'] = 999999

    # Уберем слишком дорогие товары
    expensive_criterion = 100
    expensive_items_id = data_copy[data_copy['price'] > expensive_criterion].item_id.tolist()
    data.loc[data['item_id'].isin(expensive_items_id), 'item_id'] = 999999   
    
    del data_copy

    return data

def postfilter_items(user_id, recommednations):
    pass