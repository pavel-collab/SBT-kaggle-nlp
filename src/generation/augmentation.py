from transformers import pipeline
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_label', type=str, default="Linear Algebra", help='targer class that nn will generate')
parser.add_argument('-d', '--data_path', help='path to train data')
parser.add_argument('-n', '--n_samples', type=int, default=10, help='number of generated samples')
parser.add_argument('-o', '--output_path', type=str, default='./data/', help='path to directory where we will save output file with generated data')
args = parser.parse_args()

target_label = args.target_label

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

def main():

    if args.data_path is None or args.data_path == "":
        print("ERROR you didn't set up data file path")
        return
    
    data_path = Path(args.data_path)
    assert(data_path.exists())

    paraphraser = pipeline("text2text-generation", 
                        model="ramsrigouthamg/t5_paraphraser", 
                        tokenizer="t5-base")

    prompt = "paraphrase: {text} </s>"

    df = pd.read_csv(data_path.absolute())

    for idx, row in df.iterrows():
        if idx == args.n_samples:
            break
        
        outputs = paraphraser(prompt.format(text = row['Question']), 
                            max_length=128, 
                            num_return_sequences=1, #! number of perefrase generations 
                            do_sample=True,
                            temperature=1.1,
                            top_k=500,
                            top_p=0.95)
        
        output_file_path = Path(args.output_path)
        output_file = Path(f"{output_file_path.absolute()}/{target_label.replace(' ', '-')}_augmented_samples.csv")
        file_create = output_file.exists()

        for out in outputs:
            generated = out['generated_text']
            
            # clean result before insert in file
            cleaned_generation = generated.replace(',', '').replace('\n', ' ')
            
            with open(output_file.absolute(), 'a') as fd:
                if not file_create:
                    fd.write("Question,label\n")
                fd.write(f"{cleaned_generation},{label2idx[target_label]}\n")
        
if __name__ == '__main__':
    main()
