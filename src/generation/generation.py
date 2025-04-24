from vllm import LLM, SamplingParams
from transformers import pipeline
import gc
import torch

# Загружаем LLM
# llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct')
llm = LLM(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
          tokenizer_mode="auto",
          dtype="auto",
          gpu_memory_utilization=0.9,
          max_num_seqs=128,
          max_model_len=2048)

target_label = "Linear Algebra"
labels = ["Algebra", "Geometry and Trigonometry", "Calculus and Analysis",
          "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math",
          "Linear Algebra", "Abstract Algebra and Topology"]

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
#! мы можем использовать здесь предобученную нами модель
#! ради экономии видеопамяти эту часть можно вынести в отдельный скрипт
#! мы можем попробовать здесь другие модели
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")
result = classifier(generated, labels)

print(f"\nClassification: {result['labels'][0]} (score={result['scores'][0]:.2f})")

if result['labels'][0] == target_label and result['scores'][0] > 0.7:
    print("Example accepted")
else:
    print("Example declined")