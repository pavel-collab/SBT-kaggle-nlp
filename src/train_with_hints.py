import torch
from transformers import BertConfig, Trainer, TrainingArguments, BertTokenizer, IntervalStrategy
from src.models.bert_with_attention_hints import BertWithAttentionHints
from src.data_preprocessing.dataset_with_hints import MathTextWithAttentionHints, DataCollatorWithHintMask

# Example data and configuration
texts = ["Example text 1", "Example text 2"]
labels = [0, 1]
important_words_dict = {"label_0": ["example", "text"], "label_1": ["another", "sample"]}
label2str = {0: "label_0", 1: "label_1"}

# Initialize tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = MathTextWithAttentionHints(texts, labels, tokenizer, important_words_dict, label2str)

# Split dataset into train and validation
train_dataset = dataset  # Replace with actual split logic
val_dataset = dataset  # Replace with actual split logic

# Initialize model configuration and model
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
model = BertWithAttentionHints(config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithHintMask(),
    tokenizer=tokenizer,
)

# Train the model
trainer.train()