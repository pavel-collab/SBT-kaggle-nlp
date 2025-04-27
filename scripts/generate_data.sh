#!/bin/bash

# Функция для отображения справки
usage() {
    echo "Usage: $0 [-a] [-b] [-h]"
    echo "  -a    Using data augmentation"
    echo "  -g    Usage data generation"
    echo "  -h    help"
}

# По умолчанию
generate_data() {
    # python3 ./src/generation/generation.py -t "Linear Algebra" \
    #     -n 2002 \
    #     -m ./results/bert-base-uncased_results/checkpoint-5000 \
    #     -o ./data/generated
    
    # echo "SLEEPING FOR A 10 SEC TO REST RESOURCES"
    # sleep 10
    
    # python3 ./src/generation/generation.py -t "Abstract Algebra and Topology" \
    #     -n 2006 \
    #     -m ./results/bert-base-uncased_results/checkpoint-5000 \
    #     -o ./data/generated

    # echo "SLEEPING FOR A 10 SEC TO REST RESOURCES"
    # sleep 10

    python3 ./src/generation/generation.py -t "Calculus and Analysis" \
        -n 1249 \
        -m ./results/bert-base-uncased_results/checkpoint-5000 \
        -o ./data/generated

    echo "SLEEPING FOR A 10 SEC TO REST RESOURCES"
    sleep 10

    python3 ./src/generation/generation.py -t "Probability and Statistics" \
        -n 1784 \
        -m ./results/bert-base-uncased_results/checkpoint-5000 \
        -o ./data/generated
}

augment_data() {
    python3 ./src/generation/augmentation.py -t "Linear Algebra" \
        -n 2002 \
        -d ./data/train.csv \
        -o ./data/generated

    echo "SLEEPING FOR A 10 SEC TO REST RESOURCES"
    sleep 10

    python3 ./src/generation/augmentation.py -t "Abstract Algebra and Topology" \
        -n 2006 \
        -d ./data/train.csv \
        -o ./data/generated

    echo "SLEEPING FOR A 10 SEC TO REST RESOURCES"
    sleep 10

    python3 ./src/generation/augmentation.py -t "Calculus and Analysis" \
        -n 1249 \
        -d ./data/train.csv \
        -o ./data/generated

    echo "SLEEPING FOR A 10 SEC TO REST RESOURCES"
    sleep 10

    python3 ./src/generation/augmentation.py -t "Probability and Statistics" \
        -n 1784 \
        -d ./data/train.csv \
        -o ./data/generated
}

# Обработка флагов
while getopts "agh" option; do
    case $option in
        a)
            echo "DATA AUGMENTATION"
            augment_data
            ;;
        g)
            echo "DATA GENERATION"
            generate_data
            ;;
        h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

# Если никаких флагов не передано, выполняем действие по умолчанию
if [ $OPTIND -eq 1 ]; then
    echo "DATA AUGMENTATION"
    generate_data()
fi

python3 ./src/generation/make_new_dataset.py -d ./data/generated