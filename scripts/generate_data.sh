#!/bin/bash

python3 ./src/generation/generation.py -t "Linear Algebra" \
    -n 2002 \
    -m ./results/bert-base-uncased_results/checkpoint-5000 \
    -o ./data/generated
python3 ./src/generation/augmentation.py -t "Abstract Algebra and Topology" \
    -n 2006 \
    -d ./data/train.csv \
    -o ./data/generated

python3 ./src/generation/make_new_dataset.py -d ./data/generated