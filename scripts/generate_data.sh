#!/bin/bash

#TODO: сделать константы или ассоциативный массив для меток и количества генераций
#TODO: написать 2 ветки генерации и аугментации для всех классов, выбирать ветку по флагу в скрипте, по дефолту выбирать генерацию

python3 ./src/generation/generation.py -t "Linear Algebra" \
    -n 2002 \
    -m ./results/bert-base-uncased_results/checkpoint-5000 \
    -o ./data/generated

python3 ./src/generation/augmentation.py -t "Abstract Algebra and Topology" \
    -n 2006 \
    -d ./data/train.csv \
    -o ./data/generated

python3 ./src/generation/generation.py -t "Calculus and Analysis" \
    -n 1249 \
    -m ./results/bert-base-uncased_results/checkpoint-5000 \
    -o ./data/generated

python3 ./src/generation/generation.py -t "Probability and Statistics" \
    -n 1784 \
    -m ./results/bert-base-uncased_results/checkpoint-5000 \
    -o ./data/generated

python3 ./src/generation/make_new_dataset.py -d ./data/generated