import pandas as pd
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='path to the directory with generated and augmented data')
args = parser.parse_args()

data_path = Path(args.data_path)
assert(data_path.exists())

# Список для хранения DataFrame'ов
dataframes = []

# Проходим по всем файлам в папке
for filename in os.listdir(data_path.absolute()):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path.absolute(), filename)
        # Читаем CSV файл и добавляем его в список
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Объединяем все DataFrame'ы в один
combined_df = pd.concat(dataframes, ignore_index=True)

result_file_path = os.path.join(data_path.absolute(), 'generated_train.csv')

# Сохраняем объединенные данные в новый CSV файл
combined_df.to_csv(result_file_path, index=False)

# Выводим информацию о результате
print(f'Объединено {len(dataframes)} файлов. Общая длина: {len(combined_df)} строк.')
