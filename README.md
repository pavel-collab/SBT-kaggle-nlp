```
kaggle competitions download -c classification-of-math-problems-by-kasut-academy
```

```
mkdir images
python3 ./src/run.py
./scripts/evaluate_models.sh
python3 ./src/predict.py -m ./results/...
```
