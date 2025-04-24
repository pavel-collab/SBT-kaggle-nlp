import openai
from transformers import pipeline
import os

openai.api_key = os.environ.get('OPENAI_TOKEN')

target_label = "Linear Algebra"
labels = ["Algebra", "Geometry and Trigonometry", "Calculus and Analysis",
                "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math",
                "Linear Algebra", "Abstract Algebra and Topology"]

# Промпт с уточнением стиля
prompt = f"""Generate an example of a short piece of mathematical text, which is {target_label} problem.
Don't use general phrases, give a strict, formal definition of a mathematical object."""

def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="qwen/qwen2.5-coder-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# Классификатор-судья (можно заменить на свою модель)
#! here we can put a pretrained by us basic model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
generated = generate_text(prompt)
result = classifier(generated, labels)

print(f"\nGenerated text:\n{generated}")
print(f"\nClassified as: {result['labels'][0]} (score={result['scores'][0]:.2f})")

if result['labels'][0] == target_label and result['scores'][0] > 0.7:
    print("Example accepted")
else:
    print("Example declined")