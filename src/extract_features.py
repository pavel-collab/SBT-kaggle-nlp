from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from utils.utils import get_train_data
import json

train_dataset, _ = get_train_data(use_generation=True)

texts, labels = train_dataset['text'], train_dataset['label']

idx2labels ={
    0: "Algebra", 
    1: "Geometry and Trigonometry", 
    2: "Calculus and Analysis",
    3: "Probability and Statistics", 
    4: "Number Theory", 
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra", 
    7: "Abstract Algebra and Topology"
}

# 1. Векторизация текста
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    analyzer='word'  # можно поменять на 'char_wb', если хочется символы
)
X = vectorizer.fit_transform(texts)

# 2. Быстрая логистическая регрессия
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# 3. Получение признаков
feature_names = vectorizer.get_feature_names_out()

results = {}

# Для каждого класса выводим топ-10 самых влиятельных слов
for i, class_label in enumerate(model.classes_):
    top10 = np.argsort(model.coef_[i])[-10:]
    results[idx2labels[class_label]] = [feature_names[idx] for idx in reversed(top10)]
    print(f"\nClass: {class_label} ({idx2labels[class_label]})")
    for idx in reversed(top10):
        print(f"{feature_names[idx]} : {model.coef_[i][idx]:.4f}")
        
# Запись результатов в JSON файл
with open('top_features.json', 'w') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)