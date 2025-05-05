from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, BertConfig, AutoTokenizer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import numpy as np
from utils.utils import get_train_data, get_device, get_dataset
import json
from utils.hybrid_model import HybridModelHF, BertWithHints, BertWithAttentionHints
from utils.custom_text_datset import CustomTextDataset, MathTextDataset, MathTextWithAttentionHints
from utils.constants import *
from utils.custom_trainer import CustomTrainer
from sklearn.model_selection import StratifiedKFold
from utils.datacollarator import DataCollatorWithHints, DataCollatorWithHintMask

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
#TODO: need to refactor
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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(idx2labels))
#TODO: set hint features according to the data from json file
model = BertWithAttentionHints(config) #! change hint features number

# Чтение данных из JSON файла
with open('top_features.json', 'r') as json_file:
    important_words = json.load(json_file)
    
for fold, (train_idx, val_idx) in enumerate(skf.split(X=texts, y=labels)):
    print(f"FOLD {fold}/{num_folds}")

    train_texts = [texts[idx] for idx in train_idx]
    train_labels = [labels[idx] for idx in train_idx]
    val_texts  = [texts[idx] for idx in val_idx]
    val_labels = [labels[idx] for idx in val_idx]

    train_dataset = MathTextWithAttentionHints(train_texts, train_labels, tokenizer, important_words, idx2labels)
    val_dataset = MathTextWithAttentionHints(val_texts, val_labels, tokenizer, important_words, idx2labels)

    # === Аргументы и Trainer ===
    #TODO: refactor trainer arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithHintMask(),
        tokenizer=tokenizer,
    )
    
    trainer.train()

tokenizer.save_pretrained(f"./results/tokenizer")