
import pandas as pd
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='path to the directory with generated and augmented data')
args = parser.parse_args()

data_path = Path(args.data_path)
assert(data_path.exists())

result_file_path = os.path.join(data_path.absolute(), 'generated_train.csv')

# Если файл результата уже существует, удаляем его
if os.path.exists(result_file_path):
    os.remove(result_file_path)

n_dataframes = 0

# Проходим по всем файлам в папке
for filename in os.listdir(data_path.absolute()):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path.absolute(), filename)
        
        for chunk in pd.read_csv(file_path, chunksize=1000, lineterminator='\n'):
            chunk.to_csv(result_file_path, mode='a', header=not os.path.exists(result_file_path), index=False)
        
        n_dataframes += 1
        
print(f'Объединено {n_dataframes} файлов.')
