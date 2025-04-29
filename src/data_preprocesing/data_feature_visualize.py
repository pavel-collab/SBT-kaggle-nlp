import pandas as pd
import re
import emoji
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='set path to data research .csv file')
args = parser.parse_args()

data_path = Path(args.data_path)
assert(data_path.exists())

sns.set(style="whitegrid")

def overview_report(texts, top_n=20, save_path=None):
    html_pattern = re.compile(r'<[^>]+>')
    markdown_pattern = re.compile(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)')
    latex_pattern = re.compile(r'(\\begin\{.*?\}|\\end\{.*?\}|\$.*?\$|\$\$.*?\$\$)')
    emoji_pattern = emoji.get_emoji_regexp()

    special_counts = {"HTML": 0, "Markdown": 0, "LaTeX": 0, "Emoji": 0, "None": 0}
    token_counter = Counter()
    emoji_counter = Counter()
    length_list = []
    word_lengths = []

    for text in tqdm(texts):
        if not isinstance(text, str):
            continue

        length_list.append(len(text))

        has_html = bool(html_pattern.search(text))
        has_md = bool(markdown_pattern.search(text))
        has_latex = bool(latex_pattern.search(text))
        has_emoji = bool(emoji_pattern.search(text))

        if has_html:
            special_counts["HTML"] += 1
        elif has_md:
            special_counts["Markdown"] += 1
        elif has_latex:
            special_counts["LaTeX"] += 1
        elif has_emoji:
            special_counts["Emoji"] += 1
        else:
            special_counts["None"] += 1

        tokens = re.findall(r'\w+|\S', text)
        token_counter.update(tokens)

        for token in tokens:
            if emoji_pattern.match(token):
                emoji_counter[token] += 1
            if token.isalpha():
                word_lengths.append(len(token))

    # Plot 1: Length distribution
    plt.figure(figsize=(14, 6))
    sns.histplot(length_list, bins=50, kde=True)
    plt.title("Distribution of Text Lengths")
    plt.xlabel("Length (characters)")
    plt.ylabel("Count")
    
    if not save_path:
        plt.show()
    else:
        plt.savefig(f"{save_path}/Length_distribution.png")

    # Plot 2: Top-N tokens
    plt.figure(figsize=(14, 6))
    most_common_tokens = token_counter.most_common(top_n)
    tokens, counts = zip(*most_common_tokens)
    sns.barplot(x=list(counts), y=list(tokens))
    plt.title(f"Top {top_n} Most Frequent Tokens")
    plt.xlabel("Frequency")
    plt.ylabel("Token")
    
    if not save_path:
        plt.show()
    else:
        plt.savefig(f"{save_path}/Top_n_tokens.png")

    # Plot 3: Text feature type distribution
    plt.figure(figsize=(7, 5))
    labels = list(special_counts.keys())
    sizes = list(special_counts.values())
    sns.barplot(x=sizes, y=labels)
    plt.title("Text Feature Type Distribution")
    plt.xlabel("Number of Texts")
    plt.ylabel("Feature Type")
    
    if not save_path:
        plt.show()
    else:
        plt.savefig(f"{save_path}/feature_types.png")

    # Plot 4: Emoji frequency
    if emoji_counter:
        plt.figure(figsize=(14, 6))
        most_common_emojis = emoji_counter.most_common(top_n)
        emojis, counts = zip(*most_common_emojis)
        sns.barplot(x=list(counts), y=list(emojis))
        plt.title(f"Top {top_n} Most Frequent Emojis")
        plt.xlabel("Frequency")
        plt.ylabel("Emoji")

        if not save_path:
            plt.show()
        else:
            plt.savefig(f"{save_path}/emoji_frequency.png")

    # Plot 5: Word length distribution
    if word_lengths:
        plt.figure(figsize=(14, 6))
        sns.histplot(word_lengths, bins=30, kde=False)
        plt.title("Distribution of Word Lengths")
        plt.xlabel("Word Length (characters)")
        plt.ylabel("Count")
        
        if not save_path:
            plt.show()
        else:
            plt.savefig(f"{save_path}/word_len_distrib.png")

    print("Overview complete.")
    
    
df = pd.read_csv(data_path.absolute())
df = df.rename(columns={'Question': 'text'})
# Применить к своему датасету:
overview_report(df["text"], save_path='./images')