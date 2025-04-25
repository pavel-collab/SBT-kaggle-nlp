from transformers import pipeline
import pandas as pd

paraphraser = pipeline("text2text-generation", 
                       model="ramsrigouthamg/t5_paraphraser", 
                       tokenizer="t5-base")

prompt = "paraphrase: {text} </s>"

df = pd.read_csv('../../data/train.csv')

for idx, row in df.iterrows():
    outputs = paraphraser(prompt.format(text = row['Question']), 
                          max_length=128, 
                          num_return_sequences=3, 
                          do_sample=True,
                          temperature=1.1,
                          top_k=500,
                          top_p=0.95)
    
    print(f"Original text {row['Question']}")
    print("Perephrased outputs:")
    for i, out in enumerate(outputs, 1):
        print(f"{i}) {out['generated_text']}")
    print()
    
    if idx == 5: break