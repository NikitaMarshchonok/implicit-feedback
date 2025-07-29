# app/main.py

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException

# download save model
with open('model_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
user2idx = bundle['user2idx']
item2idx = bundle['item2idx']
user_item = bundle['user_item']

# обратный маппинг индекса → оригинальный item_id
idx2item = {v:k for k,v in item2idx.items()}

app = FastAPI(title="Recommender API")

# --- добавили health-check ---
@app.get("/")
def health_check():
    return {"status": "ok"}
# ------------------------------

@app.get("/recommend/{user_id}")
def recommend(user_id: int, N: int = 10):
    if user_id not in user2idx:
        raise HTTPException(status_code=404, detail="User not found")
    uidx = user2idx[user_id]
    n_items = user_item.shape[1]
    all_items = np.arange(n_items)
    scores = model.predict(uidx, all_items)
    seen = set(user_item[uidx].indices)
    ranked = np.argsort(-scores)
    rec_idxs = [i for i in ranked if i not in seen][:N]
    # Приводим каждый элемент к обычному int
    recs = [int(idx2item[i]) for i in rec_idxs]
    return {"user_id": user_id, "recommendations": recs}
