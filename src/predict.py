from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, BertConfig
import argparse
from pathlib import Path
import torch
import numpy as np
from utils.utils import get_test_data
import pandas as pd
from utils.constants import *
from transformers import BertTokenizer
from utils.hybrid_model import HybridModelHF, BertWithHints, BertWithAttentionHints
from utils.custom_text_datset import InferenceDataset, InferenceDatasetWithHints, InferenceDatasetWithAttentionHints
import json
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

SUBMISSION_FILE_NAME = 'submission.csv'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-t', '--tokenizer_path', type=str, default=None, help='set path to saved tokenizer')
parser.add_argument('-u', '--use_hybrid', action='store_true', help='flag, if we are using hybrid model')
args = parser.parse_args()

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


model_file_path = Path(args.model_path)
assert(model_file_path.exists())

if args.tokenizer_path is None:
    tokenizer_file_path = Path(f"{model_file_path.parent.absolute()}/tokenizer")
else:
    tokenizer_file_path = Path(args.tokenizer_path)
assert(tokenizer_file_path.exists())

model_name = model_file_path.parent.name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.use_hybrid:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(idx2labels))
    #TODO: set hint features according to the data from json file
    model = BertWithAttentionHints(config=config) #! change hint features number
    
    state_dict = load_file(f"{model_file_path.absolute()}/model.safetensors")
    model.load_state_dict(state_dict=state_dict)
else:
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
    # Загрузка модели
    model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute(), num_labels=n_classes)

model.to(device)
model.eval()

test_dataset = get_test_data()

if args.use_hybrid:    
    # Чтение данных из JSON файла
    with open('top_features.json', 'r') as json_file:
        important_words = json.load(json_file)
    
    test_texts = test_dataset['text']
    # tokenized_test_dataset = InferenceDatasetWithHints(test_texts, tokenizer, important_words, idx2labels)
    tokenized_test_dataset = InferenceDatasetWithAttentionHints(test_texts, tokenizer, important_words)
else:
    # Токенизация данных
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

loader = DataLoader(tokenized_test_dataset, batch_size=16)

# === Предсказание ===
preds = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        hint_features = batch['hint_features'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, hint_features=hint_features)
        logits = outputs['logits']
        batch_preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        preds.extend(batch_preds.tolist())

df = pd.DataFrame(preds, columns=['label'])

submit_df = pd.read_csv(test_csv_file)
submit_df = submit_df.drop(columns=['Question'])
submit_df = pd.concat([submit_df, df], axis=1)

submit_df.to_csv(SUBMISSION_FILE_NAME, index=False)
print("Submission file created successfully!")