from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, BertConfig
import argparse
from pathlib import Path
from utils.constants import *
from utils.utils import (get_device, get_train_data,
                         evaleate_model, plot_confusion_matrix)
from transformers import BertTokenizer
import json
from safetensors.torch import load_file
from utils.model import BertWithFocalLoss

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

if args.output is not None and args.output != "":
    output_file_path = Path(args.output)
else: 
    output_file_path = None

model_file_path = Path(args.model_path)
assert(model_file_path.exists())

# if args.tokenizer_path is None:
#     tokenizer_file_path = Path(f"{model_file_path.parent.absolute()}/tokenizer")
# else:
#     tokenizer_file_path = Path(args.tokenizer_path)
# assert(tokenizer_file_path.exists())

model_name = model_file_path.parent.name

# детектируем девайс
device = get_device()

if args.use_hybrid:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(idx2labels))
    #TODO: set hint features according to the data from json file
    model = BertWithFocalLoss(config=config)
    
    # set model pretrained weights
    state_dict = load_file(f"{model_file_path.absolute()}/model.safetensors")
    model.load_state_dict(state_dict=state_dict)
# else:
#     # Загрузка токенизатора
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
#     # Загрузка модели
#     model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute(), num_labels=n_classes)

model.to(device)

_, val_dataset = get_train_data()


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