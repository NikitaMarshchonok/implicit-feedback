# src/train_model.py
'''
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import implicit


def load_ratings():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml-100k')
    path = os.path.join(DATA_DIR, 'u.data')
    df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df.drop(columns=['timestamp'])


def build_sparse_matrix(ratings, alpha=10.0):
    """
    Построение CSR user×item матрицы
    с весами confidence = 1 + alpha * rating
    """
    # Переиндексация пользователей и фильмов
    unique_users = ratings['user_id'].unique()
    unique_items = ratings['item_id'].unique()
    user2idx = {u:i for i,u in enumerate(unique_users)}
    item2idx = {i:j for j,i in enumerate(unique_items)}

    rows = ratings['user_id'].map(user2idx)
    cols = ratings['item_id'].map(item2idx)

    # Вместо единиц используем confidence = 1 + alpha * rating
    confidence = 1.0 + alpha * ratings['rating'].astype(np.float32)

    # user × item матрица
    user_item = coo_matrix(
        (confidence, (rows, cols)),
        shape=(len(unique_users), len(unique_items))
    ).tocsr()

    return user_item, user2idx, item2idx



def train_als(user_item, factors=30, regularization=0.05, iterations=10):
    # ALS из implicit: ждёт на вход item×user
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=False
    )
    model.fit(user_item.T)  # транспонируем → item × user
    return model


def recommend_for_user(model, user_id, user2idx, item2idx, user_item, N=10):
    """
    Рекомендуем на основе скалярного произведения факторов:
    1) забираем вектор пользователя,
    2) умножаем на матрицу факторов items,
    3) убираем уже взаимодействовавшие,
    4) берём топ-N.
    """
    uidx = user2idx[user_id]
    # 1) вектор пользователя
    user_vec = model.user_factors[uidx]               # shape = (factors,)
    # 2) score = item_factors ⋅ user_vec
    scores = model.item_factors.dot(user_vec)         # shape = (n_items,)

    # 3) какие индексы уже видел пользователь
    seen = set(user_item[uidx].indices)

    # 4) сортируем по убыванию score и фильтруем seen
    ranked = np.argsort(-scores)
    recs = [idx for idx in ranked if idx not in seen][:N]

    # переводим внутренние индексы обратно в оригинальные item_id
    idx2item = {v:k for k,v in item2idx.items()}
    return [idx2item[i] for i in recs]



if __name__ == '__main__':
    print('1) Loading ratings…')
    ratings = load_ratings()

    print('2) Building user×item matrix…')
    user_item, user2idx, item2idx = build_sparse_matrix(ratings, alpha=10.0)

    print('3) Training ALS…')
    model = train_als(user_item)

    print('4) Recommendations for user 1:')
    recs = recommend_for_user(
        model,
        user_id=1,
        user2idx=user2idx,
        item2idx=item2idx,
        user_item=user_item,
        N=10
    )
    print(recs)
'''

# src/train_model.py

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM

def load_ratings():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml-100k')
    path = os.path.join(DATA_DIR, 'u.data')
    df = pd.read_csv(path, sep='\t', names=['user_id','item_id','rating','timestamp'])
    return df.drop(columns=['timestamp'])

def build_sparse_matrix(ratings, alpha=10.0):
    """
    CSR user×item матрица с confidence = 1 + alpha * rating.
    """
    users = ratings['user_id'].unique()
    items = ratings['item_id'].unique()
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {i:j for j,i in enumerate(items)}

    rows = ratings['user_id'].map(user2idx)
    cols = ratings['item_id'].map(item2idx)
    confidence = 1.0 + alpha * ratings['rating'].astype(np.float32)

    mat = coo_matrix(
        (confidence, (rows, cols)),
        shape=(len(users), len(items))
    ).tocsr()

    return mat, user2idx, item2idx

def train_lightfm(user_item, no_components=30, loss='warp', epochs=30):
    """
    Обучаем LightFM на user×item матрице.
    """
    model = LightFM(no_components=no_components, loss=loss)
    model.fit(user_item, epochs=epochs, num_threads=4)
    return model

def recommend_for_user_lfm(model, user_id, user2idx, item2idx, user_item, N=10):
    """
    Для заданного user_id выдаёт топ-N item_id.
    """
    uidx = user2idx[user_id]
    n_items = user_item.shape[1]
    all_items = np.arange(n_items)

    # Получаем score для каждой пары (user, item)
    scores = model.predict(uidx, all_items)

    # Фильтруем уже виденные
    seen = set(user_item[uidx].indices)
    ranked = np.argsort(-scores)
    rec_idxs = [i for i in ranked if i not in seen][:N]

    # Переводим обратно в оригинальные ID
    idx2item = {v:k for k,v in item2idx.items()}
    return [idx2item[i] for i in rec_idxs]

if __name__ == '__main__':
    print('1) Loading ratings…')
    ratings = load_ratings()

    print('2) Building user×item matrix…')
    user_item, user2idx, item2idx = build_sparse_matrix(ratings, alpha=10.0)

    print('3) Training LightFM…')
    lfm = train_lightfm(user_item, no_components=30, loss='warp', epochs=30)

    print('4) Recommendations for user 1 (LightFM):')
    recs = recommend_for_user_lfm(
        lfm,
        user_id=1,
        user2idx=user2idx,
        item2idx=item2idx,
        user_item=user_item,
        N=10
    )
    print(recs)
