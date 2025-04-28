import numpy as np
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             confusion_matrix, 
                             classification_report)
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
from utils.constants import *
import seaborn as sns
import re

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    validation_accuracy = accuracy_score(predictions, labels)
    validation_precision = precision_score(predictions, labels)
    validation_recall = recall_score(predictions, labels)
    validation_f1_micro = f1_score(predictions, labels, average='micro')
    validation_f1_macro = f1_score(predictions, labels, average='macro')

    return {
        'accuracy': validation_accuracy,
        'precision': validation_precision,
        'recall': validation_recall,
        'f1_micro': validation_f1_micro,
        'f1_macro': validation_f1_macro
    }
    
def evaleate_model(model, trainer, tokenized_val_dataset, device):
    model.to(device)
    model.eval()
    predictions = trainer.predict(tokenized_val_dataset)
    
    #! in some models predictions.predictions is a complex tupple, not a numpy array 
    if isinstance(predictions.predictions, tuple):
        target_predictions = predictions.predictions[0]
    else:
        target_predictions = predictions.predictions
    
    preds = np.argmax(target_predictions, axis=-1)
    true_lables = tokenized_val_dataset['label']
    cm = confusion_matrix(true_lables, preds)
    report = classification_report(true_lables, preds)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    # Вычисление взвешенной F1-меры для текущей модели
    micro_f1 = f1_score(true_lables, preds, average='micro')
    return cm, report, accuracy, micro_f1

def plot_confusion_matrix(cm, classes, model_name=None, save_file_path=None):
    """
    Plots a confusion matrix for visualizing classification performance.

    This function takes the confusion matrix and class labels to create a heatmap
    visualization. It also allows saving the plot to a file or returning it
    without saving.

    Args:
        cm (numpy.ndarray): The confusion matrix array.
        classes (list): List of class names used in the model.
        model_name (string, optional): Name of the model for naming purposes. If None,
                                        does not set a title. Defaults to None.
        save_file_path (str, optional): Path where the plot should be saved. If None,
                                         the plot is displayed but not saved. Defaults to None.

    Returns:
        str: The filename or None if no saving occurs.

    Raises:
        AssertionError: If model_name is provided but save_file_path is not set.
    """
    with plt.style.context('default'):  
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        if model_name:
            assert save_file_path is not None
            plt.title(f"Confusion Matrix for {model_name}")
        else:
            plt.title("Confusion Matrix")
        
        if save_file_path is None:
            plt.show()
        else:
            # Verify that model_name exists before saving
            assert model_name, "model_name must be provided when save_file_path is not None"
            plt.savefig(f"{save_file_path}/confusion_matrix_{model_name}.jpg")
            return f"{save_file_path}/confusion_matrix_{model_name}.jpg"
        
def fix_random_seed(seed=20):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
'''
Функция для маскирования. Мы будем маскировать математические выражениея в тексте, чтобы
помочь модели лучше распознавать контекст вокруг математических выражений и сами математические
выражеия.
'''
def clean_math_text(text):
    # Preserve mathematical notation
    text = re.sub(r'\$(.*?)\$', r' [MATH] \1 [MATH] ', text)
    text = re.sub(r'\\\w+', lambda m: ' ' + m.group(0) + ' ', text)
    return text.strip()

def mask_latex_in_text(text):
    environments = ["equation", "equation*", "align", "align*", "multline", "multline*", "eqnarray", "eqnarray*"]
    patterns = []

    for env in environments:
        patterns.append(rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}')  # безопасное построение шаблона

    # Добавим остальные форматы
    patterns += [
        r'\$\$.*?\$\$',             # $$...$$
        r'(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)'  # $...$
    ]

    combined_pattern = '|'.join(f'({p})' for p in patterns)
    matches = list(re.finditer(combined_pattern, text, re.DOTALL))

    new_text = ""
    last_idx = 0

    for i, match in enumerate(matches):
        start, end = match.span()
        new_text += text[last_idx:start]
        # new_text += f"[LATEX]"
        new_text += f"[MATH]"
        last_idx = end

    new_text += text[last_idx:]
    return new_text
    
def get_train_data(use_generation=False, get_class_weight_flag=False):
    '''
    Можно было бы сначала формировать датасет, а потом только делить его на 
    train и test. Но тут задумка в том, что в части для валидации нет сгенерированных данных.
    '''
    df = pd.read_csv(train_csv_file)
    df = df.rename(columns={'Question': 'text'})

    # df['text'] = df['text'].apply(clean_math_text)
    # df['text'] = df['text'].apply(mask_latex_in_text)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=20)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    if use_generation:
        generated_df = pd.read_csv(generated_csv_file)
        generated_df = generated_df.rename(columns={'Question': 'text'})

        train_df = pd.concat([train_df, generated_df], ignore_index=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    if get_class_weight_flag:
        clw.class_weights = get_class_weights(train_df)
    
    return train_dataset, val_dataset

def get_dataset(use_generation=False, get_class_weight_flag=False):
    df = pd.read_csv(train_csv_file)
    train_df = df.rename(columns={'Question': 'text'})
    
    if use_generation:
        generated_df = pd.read_csv(generated_csv_file)
        generated_df = generated_df.rename(columns={'Question': 'text'})

        train_df = pd.concat([df, generated_df], ignore_index=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    
    if get_class_weight_flag:
        clw.class_weights = get_class_weights(train_df)
    
    return train_dataset

def get_test_data():
    test_df = pd.read_csv(test_csv_file)
    test_df = test_df.drop(columns=['id'])
    test_df = test_df.rename(columns={'Question': 'text'})

    test_dataset = Dataset.from_pandas(test_df)
    
    return test_dataset

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def get_class_weights(train_df):
    # Подсчитать количество изображений в каждом классе для обучающего набора данных
    train_class_counts = np.zeros(n_classes)
    for idx, row in train_df.iterrows():
        label = row['label']
        train_class_counts[label] += 1
        
    # посчитаем веса для каждого класса
    class_weights = (sum(train_class_counts.tolist()) / (n_classes * train_class_counts)).tolist()
    class_weights = torch.tensor(class_weights)
    return class_weights

def print_device_info():
    print(f"[DEBUG] Torch sees ", torch.cuda.device_count(), 'GPU(s)')
    print(f"[DEBUG] Accelerate is using device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    print()