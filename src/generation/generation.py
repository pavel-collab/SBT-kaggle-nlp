from vllm import LLM, SamplingParams
from transformers import pipeline
import gc
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pipeline', action='store_true', help='use raw pipeline to classify generationed result')
args = parser.parse_args()

# Загружаем LLM
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct')
# llm = LLM(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
#           tokenizer_mode="auto",
#           dtype="auto",
#           gpu_memory_utilization=0.9,
#           max_num_seqs=128,
#           max_model_len=2048)

target_label = "Linear Algebra"
labels = ["Algebra", "Geometry and Trigonometry", "Calculus and Analysis",
          "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math",
          "Linear Algebra", "Abstract Algebra and Topology"]

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

# Промпт с уточнением стиля
prompt = f"""Generate an example of a short piece of mathematical text, which is {target_label} problem.
Don't use general phrases, give an example of mathematical problem that teacher can give in school or univercity."""

sampling_params = SamplingParams(
            temperature=0.9, top_k=500, top_p=0.9, max_tokens=512, n=1
        )

# Генерация
outputs = llm.generate(prompt, sampling_params)
generated = outputs[0].outputs[0].text.strip()

print("\nGenerated text:")
print(generated)

# освободим память перед применением новой нейронки
gc.collect()
torch.cuda.empty_cache()

# Проверка классификатором
#! ради экономии видеопамяти эту часть можно вынести в отдельный скрипт
#! мы можем попробовать здесь другие модели
if args.pipeline:
    classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")
    result_info = classifier(generated, labels)
    score = result_info['scores'][0]
    result = result_info['labels'][0]
else:
    tokenizer_file_path = '../../results/bert-base-uncased_results/tokenizer'
    model_file_path = '../../results/bert-base-uncased_results/checkpoint-5000'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path)
    # Загрузка модели
    model = AutoModelForSequenceClassification.from_pretrained(model_file_path)
    
    # Токенизация данных
    inputs = tokenizer(generated, return_tensors='pt')

    # Прогон текста через модель
    with torch.no_grad():  # Отключаем градиенты для оптимизации
        outputs = model(**inputs)

    # Извлечение предсказаний
    logits = outputs.logits
    result_lable = torch.argmax(logits, dim=-1)
    propabilities = F.softmax(logits, dim=1)
    score = propabilities[0][result_lable].detach().item()
    result = idx2labels[result_lable.detach().item()]

print(f"\nClassification: {result} (score={score:.2f})")

if result == target_label and score > 0.7:
    print("Example accepted")
else:
    print("Example declined")