#! /bin/bash

# Указываем путь к CSV файлу
csv_file="model_evaluation_result.csv"
# Значение по умолчанию для k
default_k=12

# Функция для получения топ-k моделей
get_top_models() {
    local k="$1"
    local csv_file="$2"

    # Проверяем, существует ли файл
    if [[ ! -f "$csv_file" ]]; then
        echo "Файл $csv_file не найден!"
        exit 1
    fi

    # Читаем CSV файл, сортируем по оценке и выводим топ-k моделей
    echo "Топ $k моделей с самой высокой оценкой:"
    awk -F',' 'NR > 1 {print $1 "," $2}' "$csv_file" | sort -t',' -k2 -nr | head -n "$k"
}

# Функция для получения топ-k худших моделей с самой низкой оценкой
get_worst_models() {
    local k="$1"
    local csv_file="$2"

    # Проверяем, существует ли файл
    if [[ ! -f "$csv_file" ]]; then
        echo "Файл $csv_file не найден!"
        exit 1
    fi

    # Читаем CSV файл, сортируем по оценке и выводим топ-k худших моделей с оценками
    echo "Топ $k худших моделей с самой низкой оценкой:"
    awk -F',' 'NR > 1 {print $1 "," $2}' "$csv_file" | sort -t',' -k2 -n | head -n "$k"
}

for dir in ./results/*; do
    for model_path in $dir/*; do
        # Извлекаем родительскую директорию
        parent_directory="${model_path%/*}"
        # Извлекаем имя родительской директории
        model_name="${parent_directory##*/}"

        if [[ "$model_path" != *checkpoint-2000* ]]; then
            continue
        fi

        echo "Evaluate $model_name"
        python3 ./src/evaluate_model.py -m $model_path -o $csv_file
    done
done

# Проверяем аргументы командной строки
if [[ $# -gt 0 ]]; then
    k="$1"
else
    k="$default_k"
fi

# Вызываем функцию
get_top_models "$k" "$csv_file"

echo ""

get_worst_models "$k" "$csv_file"