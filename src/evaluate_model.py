from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import argparse
from pathlib import Path
from utils.constants import *
from utils.utils import (get_device, get_train_data,
                         evaleate_model, plot_confusion_matrix)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
args = parser.parse_args()

model_file_path = Path(args.model_path)
assert(model_file_path.exists())

#! пофиксить баг: импортировать токенайзер из отдельной дериктории
#! написать extract_model_name и передавать имя модели в имена файлов с сохраненными результатами
# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute(), num_labels=n_classes)

# детектируем девайс
device = get_device()
model.to(device)

_, val_dataset = get_train_data()

# Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

baseline_trainer = Trainer(
    model=model,
    eval_dataset=tokenized_val_dataset,
)

# cm, _, accuracy, micro_f1 = evaleate_model(model, tokenized_val_dataset, device)
cm, validation_report, accuracy, micro_f1 = evaleate_model(model, baseline_trainer, tokenized_val_dataset, device)
print("Metrics for current model:")
print(f'Test accuracy: {accuracy:.4f}')
print(validation_report)
print(f'Test F1 micro: {micro_f1:.4f}')
plot_confusion_matrix(cm, classes=range(len(classes_list)), model_name="", save_file_path='./images')