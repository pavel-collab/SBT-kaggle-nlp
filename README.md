```
kaggle competitions download -c classification-of-math-problems-by-kasut-academy
```

установка зависимостей
```
# создаем виртуальное окружение
python3 -m venv ./.venv
pip install -r requirements.txt
```

## Порядок запуска

Для начала, если мы хотим сгенерировать синтетические данные, запустим скрипт generate_data.sh с опцией -g для генерации и с опцией -a для 
аугментации.
```
./scripts/generate_data.sh -g
# или ./scripts/generate_data.sh -a
```

По итогу работы этого сприпта в дериктории data появится поддериктория generated с несколькими cvs файлами. В файлах будут находится 
сгенерированные данные для каждого класса. Кроме того, будет файл generated_train.csv, который объединяет все генерации. По факту, этот файл
будет являться дополнением к тренеровочному датасету.

Далее запускаем обучение моделей. При запуске обучения можно выбрать опцию --use_generation, тогда при обучении будут задействованы сгенерированные
данные.
```
python3 ./src/run.py --use_generation
# или тренеровка без учета сгенерированных данных python3 ./src/run.py
```

Поскольку обучаются сразу несколько разных моделей, это может занять некоторое время. В итоге натренированные модели можно будет найти в 
дериктории results, а логи обучения в logs. Для просмотра логов обучения через tensorboard
```
tensorboard --logdir=./logs
```

После того, как модели обучились, можно прогнать валидацию на сохраненных моделях. Для этого запускаем скрипт
```
mkdir images
./scripts/evaluate_models.sh
```

Он прогонит все сохраненные модели на валидационном датасете, выведет результаты и оценки валидации и сохранит графики с матрицами конволюции для
каждой модели в images.

Наконец, чтобы сделать предсказание, запустим скрипт
```
python3 ./src/predict.py -m ./results/bert-base-uncased_results/checkpoint-3000
```
в параметре -m указываем путь к сохраненной модели, которую собираемся использовать для предсказания.

Submit result
```
kaggle competitions submit -c classification-of-math-problems-by-kasut-academy -f submission.csv -m "Message"
```

### accelerate config
```
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: NO
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
mixed_precision: fp16  # no или fp16 / bf16, если хочешь ускорить
use_cpu: false
same_network: true
```

accelerate run function
```
accelerate launch ./src/run.py --use_generation
```

### Препроцессниг данных

Перед препроцессингом хорошо бы исследовать данные на то, какие необычные ньюансы в них могут содержаться. Есть ли там html, 
latex, ссылки, email адреса, какая средняя длина токенов и т д. Для этого приготовлены 2 специальных скрипта
```
python3 ./src/data_preprocesing/check_data.py -d ./data/train.csv
```
выведет общую информацию о датасете.
```
python3 ./src/data_preprocesing/data_feature_visualize.py -d ./data/train.csv
```
сделает графики с информацией о специфических фичах в данных.

### Применение TF-IDF

У нас представлен простой классификатор на основе tf-idf и логистической регрессии. Такой классификатор может служить в качестве baseline 
модели, а так же показывает, на сколько сложна задача и какой уровень точности можно ожидать от классификатора.

Кроме того tf-idf помогает выделить наиболее релевантные фичи в текстах для каждого класса.