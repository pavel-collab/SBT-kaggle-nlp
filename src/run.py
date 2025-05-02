import argparse
from utils.constants import *
from utils.utils import (fix_random_seed,
                         get_device,
                         get_dataset,
                         print_device_info)
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          DataCollatorWithPadding,
                          Trainer, TrainingArguments, BertConfig)
from utils.custom_trainer import CustomTrainer
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import Dataset
from utils.model import BertWithFocalLoss

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
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(classes_list))
        model = BertWithFocalLoss(config=config)

        model.to(device)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X=texts, y=labels)):
            print(f"FOLD {fold}/{num_folds}")

            train_dtst = train_dataset.select(train_idx)
            val_dtst  = train_dataset.select(val_idx)
            
            # Токенизация данных
            def tokenize_function(examples):
                return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

            tokenized_train_dataset = train_dtst.map(tokenize_function, batched=True)
            tokenized_val_dataset = val_dtst.map(tokenize_function, batched=True)
            
            training_args = TrainingArguments(
                output_dir="./results",
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                num_train_epochs=3,
                logging_steps=100,
                load_best_model_at_end=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_val_dataset,
                tokenizer=tokenizer,
            )
            
            try:
                trainer.train()
            except Exception as ex:
                print(f"[ERROR] with training {model_name}: {ex}")
                
        tokenizer.save_pretrained(f"./results/{model_name.replace('/', '-')}_results/tokenizer")
except KeyboardInterrupt:
    print(f"[STOP] training with keyboard interrupt")