import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.utils import get_train_data

train_dataset, val_dataset = get_train_data(use_generation=True)

X_train, y_train = train_dataset['text'], train_dataset['label']
X_valid, y_valid = val_dataset['text'], val_dataset['label']

# TF-IDF по словам
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=10000
)

# TF-IDF по символам
char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=5000
)

# Объединение признаков
combined_features = FeatureUnion([
    ('word_tfidf', word_vectorizer),
    ('char_tfidf', char_vectorizer)
])

# Создание pipeline: сначала признаки, потом классификатор
pipeline = Pipeline([
    ('features', combined_features),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказание и отчёт
y_pred = pipeline.predict(X_valid)
print(classification_report(y_valid, y_pred))