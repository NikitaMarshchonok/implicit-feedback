# app/main.py

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException

# Загружаем сериализованный бандл с моделью и метаданными
with open('model_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
user2idx = bundle['user2idx']
item2idx = bundle['item2idx']
user_item = bundle['user_item']

# Обратный маппинг: индекс → оригинальный item_id
idx2item = {v: k for k, v in item2idx.items()}

app = FastAPI(title="Recommender API")


@app.get("/", include_in_schema=False)
async def health_check():
    """
    Root endpoint для startup probe Cloud Run.
    """
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, N: int = 10):
    if user_id not in user2idx:
        raise HTTPException(status_code=404, detail="User not found")
    uidx = user2idx[user_id]

    # Прогоняем модель по всем айтемам
    n_items = user_item.shape[1]
    all_items = np.arange(n_items)
    scores = model.predict(uidx, all_items)

    # Отбираем непросмотренные и сортируем
    seen = set(user_item[uidx].indices)
    ranked = np.argsort(-scores)
    rec_idxs = [i for i in ranked if i not in seen][:N]

    # Переводим внутренние индексы в внешние item_id и int
    recs = [int(idx2item[i]) for i in rec_idxs]

    return {"user_id": user_id, "recommendations": recs}
