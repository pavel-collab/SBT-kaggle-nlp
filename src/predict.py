from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import argparse
from pathlib import Path
import torch
import numpy as np
from utils.utils import get_test_data
import pandas as pd
from utils.constants import *

SUBMISSION_FILE_NAME = 'submission.csv'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
args = parser.parse_args()

model_file_path = Path(args.model_path)
assert(model_file_path.exists())

tokenizer_file_path = Path(f"{model_file_path.parent.absolute()}/tokenizer")
assert(tokenizer_file_path.exists())

model_name = model_file_path.parent.name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute())

model.to(device)
model.eval()

test_dataset = get_test_data()

# Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

predicted_trainer = Trainer(
    model=model,
    eval_dataset=tokenized_test_dataset,
)

with torch.no_grad():
    # Получение предсказаний
    predictions = predicted_trainer.predict(tokenized_test_dataset)

#! in some models predictions.predictions is a complex tupple, not a numpy array 
if isinstance(predictions.predictions, tuple):
    target_predictions = predictions.predictions[0]
else:
    target_predictions = predictions.predictions

preds = np.argmax(target_predictions, axis=-1)

df = pd.DataFrame(preds, columns=['label'])

submit_df = pd.read_csv(test_csv_file)
submit_df = submit_df.drop(columns=['Question'])
submit_df = pd.concat([submit_df, df], axis=1)

submit_df.to_csv(SUBMISSION_FILE_NAME, index=False)
print("Submission file created successfully!")