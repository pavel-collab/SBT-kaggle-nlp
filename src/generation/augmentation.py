from transformers import pipeline

paraphraser = pipeline("text2text-generation", 
                       model="ramsrigouthamg/t5_paraphraser", 
                       tokenizer="t5-base")

text = r"For an integer $n\geq 3$, let $\theta=2\pi/n$.  Evaluate the determinant of the $n\times n$ matrix $I+A$, where $I$ is the $n\times n$ identity matrix and $A=(a_{jk})$ has entries $a_{jk}=\cos(j\theta+k\theta)$ for all $j,k$."
prompt = f"paraphrase: {text} </s>"

outputs = paraphraser(prompt, 
                      max_length=128, 
                      num_return_sequences=3, 
                      do_sample=True,
                      temperature=1.1,
                      top_k=500,
                      top_p=0.95)

print("Perephrased outputs:")
for i, out in enumerate(outputs, 1):
    print(f"{i}) {out['generated_text']}")