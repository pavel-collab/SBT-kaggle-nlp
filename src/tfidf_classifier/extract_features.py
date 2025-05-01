from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

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

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='set path to data research .csv file')
args = parser.parse_args()

train_csv_file = Path(args.data_path)
assert(train_csv_file.exists())

df = pd.read_csv(train_csv_file.absolute())
df = df.rename(columns={'Question': 'text'})

texts, labels = df['text'], df['label']

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

# Для каждого класса выводим топ-10 самых влиятельных слов
for i, class_label in enumerate(model.classes_):
    top10 = np.argsort(model.coef_[i])[-10:]
    print(f"\nClass: {class_label} ({idx2labels[class_label]})")
    for idx in reversed(top10):
        print(f"{feature_names[idx]} : {model.coef_[i][idx]:.4f}")