# src/evaluate.py

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import implicit
from sklearn.model_selection import train_test_split

def load_ratings():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml-100k')
    path = os.path.join(DATA_DIR, 'u.data')
    df = pd.read_csv(path, sep='\t', names=['user_id','item_id','rating','timestamp'])
    return df

def build_matrix(df):
    # та же переиндексация, но на основе переданных df
    users = df['user_id'].unique()
    items = df['item_id'].unique()
    u2i = {u:i for i,u in enumerate(users)}
    i2i = {i:j for j,i in enumerate(items)}
    rows = df['user_id'].map(u2i)
    cols = df['item_id'].map(i2i)
    data = np.ones(len(df), dtype=np.float32)
    mat = coo_matrix((data, (rows, cols)),
                     shape=(len(users), len(items))).tocsr()
    return mat, u2i, i2i

def train_model(train_mat, factors=30, reg=0.05, iters=10):
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iters,
        use_gpu=False
    )
    model.fit(train_mat.T)
    return model

def precision_at_k(model, train_mat, test_df, u2i, i2i, K=10):
    """
    Для каждого пользователя в test_df считаем:
      Precision@K = |рекомендованные ∩ реальные| / K
    """
    # Группируем тест по пользователям
    test_by_user = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    precisions = []

    for user, true_items in test_by_user.items():
        # если пользователь был в train
        if user not in u2i:
            continue
        uidx = u2i[user]
        # получаем топ-K рекомендаций
        scores = model.item_factors.dot(model.user_factors[uidx])
        ranked = np.argsort(-scores)
        # фильтруем уже виденные (из train)
        seen = set(train_mat[uidx].indices)
        recs = [i for i in ranked if i not in seen][:K]
        # переводим индексы обратно в item_id
        idx2item = {v:k for k,v in i2i.items()}
        rec_items = {idx2item[i] for i in recs}
        # precision
        precisions.append(len(rec_items & true_items) / K)

    return np.mean(precisions)

if __name__ == '__main__':
    # 1) Загрузка всех рейтингов
    ratings = load_ratings()

    # 2) Делим на train/test (20% по случайности)
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    # 3) Строим матрицу train
    train_mat, u2i, i2i = build_matrix(train_df)

    # 4) Обучаем модель
    print('Training ALS on train set…')
    model = train_model(train_mat, factors=30, reg=0.05, iters=20)

    # 5) Считаем Precision@10
    prec = precision_at_k(model, train_mat, test_df, u2i, i2i, K=10)
    print(f'Precision@10 (на тесте ~20%): {prec:.4f}')

