# save_model.py

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM

# Повторяем загрузку и подготовку
def load_ratings():
    DATA_DIR = os.path.join('data', 'ml-100k')
    path = os.path.join(DATA_DIR, 'u.data')
    df = pd.read_csv(path, sep='\t', names=['user_id','item_id','rating','timestamp'])
    return df.drop(columns=['timestamp'])

def build_matrix(ratings, alpha=10.0):
    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique()
    u2i = {u:i for i,u in enumerate(users)}
    i2i = {i:j for j,i in enumerate(items)}
    rows = ratings['user_id'].map(u2i)
    cols = ratings['item_id'].map(i2i)
    data = 1.0 + alpha * ratings['rating'].astype(np.float32)
    mat = coo_matrix((data, (rows, cols)),
                     shape=(len(users), len(items))).tocsr()
    return mat, u2i, i2i

def train_and_save(path='model_bundle.pkl'):
    ratings = load_ratings()
    user_item, u2i, i2i = build_matrix(ratings, alpha=10.0)

    model = LightFM(no_components=30, loss='warp')
    model.fit(user_item, epochs=30, num_threads=4)

    # сохраним всё в один файл
    bundle = {
        'model': model,
        'user2idx': u2i,
        'item2idx': i2i,
        'user_item': user_item
    }
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Model bundle saved to {path}")

if __name__ == '__main__':
    train_and_save()
