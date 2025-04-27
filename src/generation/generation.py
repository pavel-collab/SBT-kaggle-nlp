from vllm import LLM, SamplingParams
from transformers import pipeline
import gc
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pipeline', action='store_true', help='use raw pipeline to classify generationed result')
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-t', '--target_label', type=str, default="Linear Algebra", help='targer class that nn will generate')
parser.add_argument('-n', '--n_samples', type=int, default=10, help='number of generated samples')
parser.add_argument('-o', '--output_path', type=str, default='./data/', help='path to directory where we will save output file with generated data')
args = parser.parse_args()

# Загружаем LLM
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct',
        #   tokenizer_mode="auto",
        #   dtype="auto",
          gpu_memory_utilization=0.9,
          max_num_seqs=128,
          max_model_len=2048)
# llm = LLM(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
#           tokenizer_mode="auto",
#           dtype="auto",
#           gpu_memory_utilization=0.9,
#           max_num_seqs=128,
#           max_model_len=2048)

target_label = args.target_label
n_samples = args.n_samples

#TODO: move to constants file
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
prompt = f"""
Generate an example of a short piece of mathematical text, which is {target_label} problem.
Don't use general phrases, give an example of mathematical problem that teacher can give in school or univercity.
Make a brief answer. Only text of the unswer without introduction phrases.
"""

sampling_params = SamplingParams(
            temperature=0.9, top_k=500, top_p=0.9, max_tokens=512, n=n_samples
        )

# Генерация
outputs = llm.generate(prompt, sampling_params)

accepted_samples_number = 0
for output in tqdm(outputs[0].outputs):
    #TODO полная строка генерации outputs[0].outputs[0] -- разобраться что значат эти индексы
    #TODO в генерациях остаются артефакты генерации в начале предложения, придумать, как убрать их
    generated = output.text.strip()

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
        model_file_path = Path(args.model_path)
        assert(model_file_path.exists())

        tokenizer_file_path = Path(f"{model_file_path.parent.absolute()}/tokenizer")
        assert(tokenizer_file_path.exists())

        model_name = model_file_path.parent.name

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
        # Загрузка модели
        model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute())
        
        # Токенизация данных
        inputs = tokenizer(generated, 
                           return_tensors='pt', 
                           padding=True, 
                           truncation=True, 
                           max_length=256, 
                           add_special_tokens = True)

        # Прогон текста через модель
        with torch.no_grad():  # Отключаем градиенты для оптимизации
            outputs = model(**inputs)

        # Извлечение предсказаний
        logits = outputs.logits
        result_lable = torch.argmax(logits, dim=-1)
        propabilities = F.softmax(logits, dim=1)
        score = propabilities[0][result_lable].detach().item()
        result = idx2labels[result_lable.detach().item()]
        
    output_file_path = Path(args.output_path)
    output_file = Path(f"{output_file_path.absolute()}/{target_label.replace(' ', '-')}_generated_samples.csv")
    file_create = output_file.exists()

    if result == target_label and score > 0.7:
        # clean result before insert in file
        cleaned_generation = generated.replace(',', '').replace('\n', ' ')
        
        with open(output_file.absolute(), 'a') as fd:
            if not file_create:
                fd.write("Question,label\n")
            fd.write(f"{cleaned_generation},{result_lable.detach().item()}\n")
        accepted_samples_number += 1

gc.collect()
torch.cuda.empty_cache()
    
print("SUCCESSFUL GENERATION COMPLITION.")
print(f"NUMBER OF ACCEPTED SAMPLES [{accepted_samples_number}/{n_samples}]")