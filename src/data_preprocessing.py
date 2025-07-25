import pandas as pd
import os

# path to ratings file
DATA_DIR = os.path.join(os.getcwd(), 'data', 'ml-100k')
ratings_file = os.path.join(DATA_DIR, 'u.data')

# read : user_id, item_id, rating, timestamp
col_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(ratings_file, sep='\t', names=col_names)

# Let's remove the timestamp — it's not needed for implicit feedback.
ratings = ratings.drop(columns=['timestamp'])

# check the first 5
print(ratings.shape)
ratings.head()

# 1) Сколько уникальных пользователей и фильмов?
n_users = ratings['user_id'].nunique()
n_items = ratings['item_id'].nunique()
print(f'Пользователей: {n_users}, фильмов: {n_items}')

# 2) Распределение числа взаимодействий на пользователя
interactions_per_user = ratings.groupby('user_id').size()
print(
    'Интеракций на user (мин/ср/макс):',
    interactions_per_user.min(),
    round(interactions_per_user.mean(), 1),
    interactions_per_user.max()
)

# 3) Распределение числа взаимодействий на фильм
interactions_per_item = ratings.groupby('item_id').size()
print(
    'Интеракций на item (мин/ср/макс):',
    interactions_per_item.min(),
    round(interactions_per_item.mean(), 1),
    interactions_per_item.max()
)