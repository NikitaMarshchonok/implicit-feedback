import os, urllib.request, zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
zip_path = os.path.join(DATA_DIR, 'ml-100k.zip')

if not os.path.exists(zip_path):
    print('Скачиваем MovieLens 100K...')
    urllib.request.urlretrieve(url, zip_path)
    print('Распаковываем...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    print('Готово! Файлы в', DATA_DIR)
else:
    print('Датасет уже есть.')
