from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import numpy as np
from utils.utils import get_train_data, get_device, get_dataset
import json
from utils.hybrid_model import HybridModelHF
from utils.custom_text_datset import CustomTextDataset
from utils.constants import *
from utils.custom_trainer import CustomTrainer
from sklearn.model_selection import StratifiedKFold

#TODO: move to constants
label2idx ={
    "Algebra": 0, 
    "Geometry and Trigonometry": 1, 
    "Calculus and Analysis": 2,
    "Probability and Statistics": 3, 
    "Number Theory": 4, 
    "Combinatorics and Discrete Math": 5,
    "Linear Algebra": 6, 
    "Abstract Algebra and Topology": 7
}

device = get_device()

# === МЕТРИКИ ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

model_name = 'bert-base-uncased'

# === ПОДГОТОВКА ===
train_dataset = get_dataset(use_generation=True,
                            get_class_weight_flag=False)

# print(f"[DEBUG] class weights: {clw.class_weights}")

texts, labels = train_dataset['text'], train_dataset['label']

skf = StratifiedKFold(n_splits=num_folds, 
                      shuffle=True, 
                      random_state=42)

# === TF-IDF признаки ===
#TODO: maybe it would be better to move tf-idf feature extraction to separate script and get it for example from the file in this script
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

print(f"[DEBUG] train logestic regression to extract relevant features")
lr = LogisticRegression(max_iter=1000)
try:
    lr.fit(X, labels)
except Exception as e:
    print(f"ERROR during train logistic regression with tf-idf features: {e}")

top_n = 50
top_indices = np.argsort(np.abs(lr.coef_).max(axis=0))[-top_n:]
top_features = set(feature_names[top_indices])

# === Tokenizer и генерация extra features ===
tokenizer = BertTokenizer.from_pretrained(model_name)

#TODO: move to utils.py
def encode_data(texts, top_words):
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    extra_feats = []
    for text in texts:
        tokens = set(text.lower().split())
        feats = [1.0 if word in tokens else 0.0 for word in top_words]
        extra_feats.append(feats)
    enc['extra_features'] = torch.tensor(extra_feats, dtype=torch.float)
    return enc

model = HybridModelHF(num_labels=len(label2idx), extra_feat_dim=top_n)
model.to(device)

for fold, (train_idx, val_idx) in enumerate(skf.split(X=texts, y=labels)):
    print(f"FOLD {fold}/{num_folds}")

    train_texts = [texts[idx] for idx in train_idx]
    train_labels = [labels[idx] for idx in train_idx]
    val_texts  = [texts[idx] for idx in val_idx]
    val_labels = [labels[idx] for idx in val_idx]

    train_enc = encode_data(train_texts, top_features)
    val_enc = encode_data(val_texts, top_features)

    train_dataset = CustomTextDataset(train_enc, torch.tensor(train_labels))
    val_dataset = CustomTextDataset(val_enc, torch.tensor(val_labels))

    # === Аргументы и Trainer ===
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name.replace('/', '-')}_results_fold_{fold}",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epoches,
        weight_decay=0.01,
        logging_dir=f"./logs/{model_name.replace('/', '-')}_logs_fold_{fold}", 
        save_steps=1000, # сохранение чекпоинтов модели каждые 1000 шагов# директория для логов TensorBoard
        logging_steps=100,
        save_total_limit=5, # Сохранять только последние 5 чекпоинтов
        fp16=True,
        gradient_accumulation_steps=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,  # чтобы Trainer сохранял токенайзер
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

tokenizer.save_pretrained(f"./results/tokenizer")

with open('./results/top_features.json', 'w') as file:
    json.dump(list(top_features), file)