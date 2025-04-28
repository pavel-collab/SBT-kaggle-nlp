import argparse
from utils.constants import *
from utils.utils import (fix_random_seed,
                         get_device,
                         get_dataset,
                         print_device_info)
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          DataCollatorWithPadding,
                          Trainer, TrainingArguments)
from utils.custom_trainer import CustomTrainer
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--use_generation', action='store_true', help='if we using generation data for train')
args = parser.parse_args()

fix_random_seed()
device = get_device()

train_dataset = get_dataset(use_generation=args.use_generation,
                            get_class_weight_flag=False)

texts = train_dataset['text']
labels = train_dataset['label']

# print_device_info()

skf = StratifiedKFold(n_splits=num_folds, 
                      shuffle=True, 
                      random_state=42)

try:
    for model_name in model_list:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X=texts, y=labels)):
            print(f"FOLD {fold}/{num_folds}")

            train_dtst = train_dataset.select(train_idx)
            val_dtst  = train_dataset.select(val_idx)
            
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            data_collator = DataCollatorWithPadding(tokenizer, max_length=256, padding=True) #? нужен для чего

            model.to(device)
            
            # Токенизация данных
            def tokenize_function(examples):
                return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

            tokenized_train_dataset = train_dtst.map(tokenize_function, batched=True)
            tokenized_val_dataset = val_dtst.map(tokenize_function, batched=True)
            
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
                gradient_accumulation_steps=2,
                # load_best_model_at_end=True,          # Загружать лучшую модель в конце
                # metric_for_best_model="accuracy",     # Метрика для определения "лучшей" модели
                # greater_is_better=True,                # Указать, что большее значение метрики лучше
            )
            
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_val_dataset,
                # compute_metrics=compute_metrics,
                class_weights=clw.class_weights.to(device)
            )
            
            try:
                trainer.train()
            except Exception as ex:
                print(f"[ERROR] with training {model_name}: {ex}")
                
        tokenizer.save_pretrained(f"./results/{model_name.replace('/', '-')}_results/tokenizer")
except KeyboardInterrupt:
    print(f"[STOP] training with keyboard interrupt")