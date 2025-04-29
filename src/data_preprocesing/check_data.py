import pandas as pd
import re
import emoji
from collections import Counter
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='set path to data research .csv file')
args = parser.parse_args()

data_path = Path(args.data_path)
assert(data_path.exists())

def analyze_texts(texts):
    html_pattern = re.compile(r'<[^>]+>')
    markdown_pattern = re.compile(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)')
    latex_pattern = re.compile(r'(\\begin\{.*?\}|\\|\\|\\end\{.*?\}|\$.*?\$|\$\$.*?\$\$)')
    emoji_pattern = emoji.get_emoji_regexp()

    html_count = 0
    markdown_count = 0
    latex_count = 0
    emoji_count = 0

    token_counter = Counter()
    length_list = []

    for text in texts:
        if not isinstance(text, str):
            continue

        length_list.append(len(text))

        if html_pattern.search(text):
            html_count += 1
        if markdown_pattern.search(text):
            markdown_count += 1
        if latex_pattern.search(text):
            latex_count += 1
        if emoji_pattern.search(text):
            emoji_count += 1

        tokens = re.findall(r'\w+|\S', text)  # простая токенизация
        token_counter.update(tokens)

    total_texts = len(texts)

    stats = {
        'total_texts': total_texts,
        'avg_length': sum(length_list) / total_texts,
        'max_length': max(length_list),
        'min_length': min(length_list),
        'html_texts': html_count,
        'markdown_texts': markdown_count,
        'latex_texts': latex_count,
        'emoji_texts': emoji_count,
        'top_tokens': token_counter.most_common(20),
        'rare_tokens': [token for token, count in token_counter.items() if count == 1][:20]
    }

    return stats


df = pd.read_csv(data_path.absolute())
df = df.rename(columns={'Question': 'text'})

# допустим у тебя датафрейм df
stats = analyze_texts(df["text"])
for key, value in stats.items():
    print(f"{key}: {value}")