from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import argparse
from pathlib import Path
from utils.constants import *
from utils.utils import (get_device, get_train_data,
                         evaleate_model, plot_confusion_matrix)
from transformers import BertTokenizer
from utils.hybrid_model import HybridModelHF
from utils.custom_text_datset import CustomTextDataset
import json
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-t', '--tokenizer_path', type=str, default=None, help='set path to saved tokenizer')
parser.add_argument('-u', '--use_hybrid', action='store_true', help='flag, if we are using hybrid model')
parser.add_argument('-o', '--output', help='set a path to the output filename, programm will write a model name and final accuracy')
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

if args.output is not None and args.output != "":
    output_file_path = Path(args.output)
else: 
    output_file_path = None

model_file_path = Path(args.model_path)
assert(model_file_path.exists())

if args.tokenizer_path is None:
    tokenizer_file_path = Path(f"{model_file_path.parent.absolute()}/tokenizer")
else:
    tokenizer_file_path = Path(args.tokenizer_path)
assert(tokenizer_file_path.exists())

model_name = model_file_path.parent.name

# детектируем девайс
device = get_device()

if args.use_hybrid:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_file_path.absolute())
    model = HybridModelHF(num_labels=len(label2idx), extra_feat_dim=50, model_path_str=model_file_path.absolute())
    state_dict = load_file(f"{model_file_path.absolute()}/model.safetensors")
    model.load_state_dict(state_dict=state_dict)
else:
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
    # Загрузка модели
    model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute(), num_labels=n_classes)

model.to(device)

_, val_dataset = get_train_data()

if args.use_hybrid:
    #TODO: move to utils
    def encode_data(texts, top_words):
        enc = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        extra_feats = []
        for text in texts:
            tokens = set(text.lower().split())
            feats = [1.0 if word in tokens else 0.0 for word in top_words]
            extra_feats.append(feats)
        enc['extra_features'] = torch.tensor(extra_feats, dtype=torch.float)
        return enc
    
    with open(f"{model_file_path.parent.parent.absolute()}/top_features.json", 'r') as file:
        top_features = set(json.load(file))
    
    val_texts, val_labels = val_dataset['text'], val_dataset['label']
    val_enc = encode_data(val_texts, top_features)
    tokenized_val_dataset = CustomTextDataset(val_enc, torch.tensor(val_labels))
    true_labels = val_labels
else:
    # Токенизация данных
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)
    
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    true_labels = tokenized_val_dataset['label']

baseline_trainer = Trainer(
    model=model,
    eval_dataset=tokenized_val_dataset,
)

try:
    print(f"EVALUATE MODEL {model_name}")
    cm, validation_report, accuracy, micro_f1 = evaleate_model(model, 
                                                               baseline_trainer, 
                                                               tokenized_val_dataset, 
                                                               true_labels, 
                                                               device)
    print("Metrics for current model:")
    print(f'Test accuracy: {accuracy:.4f}')
    print(validation_report)
    print(f'Test F1 micro: {micro_f1:.4f}')
    
    plot_confusion_matrix(cm, classes=range(len(classes_list)), model_name=model_name, save_file_path='./images')
    
    if output_file_path is not None:
        file_create = output_file_path.exists()
        
        with open(output_file_path.absolute(), 'a') as fd:
            if not file_create:
                fd.write("model,accuracy\n")
            fd.write(f"{model_name},{micro_f1}\n")
except Exception as ex:
    print(f"ERROR during evaluating model {model_name}: {ex}")