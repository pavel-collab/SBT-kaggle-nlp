from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import numpy as np
from utils.utils import get_train_data

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

# === ПОДГОТОВКА ДАННЫХ ===
class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'extra_features': self.encodings['extra_features'][idx],
            'labels': self.labels[idx]
        }

# === МОДЕЛЬ ===
class HybridModelHF(nn.Module):
    def __init__(self, num_labels, extra_feat_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + extra_feat_dim, num_labels)

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        concat = torch.cat((pooled, extra_features), dim=1)
        logits = self.classifier(self.dropout(concat))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits}

# === МЕТРИКИ ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# === ПОДГОТОВКА ===
train_dataset, val_dataset = get_train_data(use_generation=True)

train_texts, train_labels = train_dataset['text'], train_dataset['label']
val_texts, val_labels = val_dataset['text'], val_dataset['label']

# === TF-IDF признаки ===

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(train_texts)
feature_names = vectorizer.get_feature_names_out()

print(f"[DEBUG] train logestic regression to extract relevant features")
lr = LogisticRegression(max_iter=1000)
lr.fit(X, train_labels)

top_n = 50
top_indices = np.argsort(np.abs(lr.coef_).max(axis=0))[-top_n:]
top_features = set(feature_names[top_indices])

# === Tokenizer и генерация extra features ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(texts, top_words):
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    extra_feats = []
    for text in texts:
        tokens = set(text.lower().split())
        feats = [1.0 if word in tokens else 0.0 for word in top_words]
        extra_feats.append(feats)
    enc['extra_features'] = torch.tensor(extra_feats, dtype=torch.float)
    return enc

train_enc = encode_data(train_texts, top_features)
val_enc = encode_data(val_texts, top_features)

train_dataset = CustomTextDataset(train_enc, torch.tensor(train_labels))
val_dataset = CustomTextDataset(val_enc, torch.tensor(val_labels))

#TODO: refactor arguments, use some constants from constants.py
# === Аргументы и Trainer ===
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_dir="./logs/"
)

model = HybridModelHF(num_labels=len(label2idx), extra_feat_dim=top_n)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,  # чтобы Trainer сохранял токенайзер
    compute_metrics=compute_metrics,
)

tokenizer.save_pretrained(f"./results/tokenizer")

trainer.train()