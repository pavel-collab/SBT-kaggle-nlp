import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import argparse
from pathlib import Path
# from ..utils.constants import train_csv_file

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='set path to data research .csv file')
args = parser.parse_args()

train_csv_file = Path(args.data_path)
assert(train_csv_file.exists())

df = pd.read_csv(train_csv_file.absolute())
df = df.rename(columns={'Question': 'text'})

train_df, val_df = train_test_split(df, test_size=0.2, random_state=20)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

X_train, y_train = train_df['text'], train_df['label']
X_valid, y_valid = val_df['text'], val_df['label']

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