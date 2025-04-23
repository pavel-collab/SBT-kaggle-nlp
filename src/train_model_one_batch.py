from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
import argparse
import torch

parser = argparse.ArgumentParser(description="Пример работы с argparse.")

# Добавление аргумента с помощью метода add_argument
parser.add_argument(
    "--model",
    type=str,
    default="distilbert-base-uncased",  # Значение по умолчанию
    help="name of testing model"
)

classes_list = ["Algebra", "Geometry and Trigonometry", "Calculus and Analysis",
                "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math",
                "Linear Algebra", "Abstract Algebra and Topology"]
n_classes = len(classes_list)

# Парсинг аргументов
args = parser.parse_args()

# детектируем девайс
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_csv_file = '../data/train.csv'
def main():
    # Загрузка токенизатора
    model_name = args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as ex:
        print(f"There are troubles during importing tokenizer for model {model_name}: {ex}")
        return
        
    try:
        # Загрузка модели
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
    except Exception as ex:
        print(f"There are troubles during importing model artifacts for model {model_name}: {ex}")
        return

    model.to(device)

    df = pd.read_csv(train_csv_file)
    df = df.rename(columns={'Question': 'text'})

    train_df, val_df = train_test_split(df, test_size=0.5, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Токенизация
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = train_dataset.map(tokenize_function, batched=True)

    # Настройка параметров обучения
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        eval_steps=1,  # Параметр для валидации
        logging_steps=1,  # Параметр для логирования
        per_device_train_batch_size=2,  # Обратите внимание на размер батча
        per_device_eval_batch_size=2,
        num_train_epochs=1,  # Установите количество эпох в 1
        max_steps=1,  # Ограничьте количество шагов до 1
        save_total_limit=1,  # Ограничьте количество сохраненных моделей
    )

    # Создание Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val, 
    )

    try:
        try:
            # Запуск обучения
            trainer.train()
            print(f"model {model_name} was trained and validated successful on one batch")
        except Exception as ex:
            print(f"There are troubles during training model {model_name}: {ex}")
    except KeyboardInterrupt:
        print("Traning model was interrupted by keyboard")
        
if __name__ == '__main__':
    main()