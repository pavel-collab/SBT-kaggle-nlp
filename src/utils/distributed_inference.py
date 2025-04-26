from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Инициализация
accelerator = Accelerator()

# Загрузка модели и токенизатора
model = AutoModelForSequenceClassification.from_pretrained("./results")  # путь к сохранённой модели
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Загружаем датасет (тестовый или пользовательский)
dataset = load_dataset("ag_news", split="test[:100]")  # ограничим до 100 примеров

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataloader = DataLoader(dataset, batch_size=32)

# Подготавливаем модель и данные
model.eval()
model, dataloader = accelerator.prepare(model, dataloader)

# Инференс
all_preds = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        preds = accelerator.gather(preds)
        all_preds.extend(preds.cpu().numpy())

# Печать результатов
for i, pred in enumerate(all_preds):
    print(f"{i+1}: Predicted class = {pred}")