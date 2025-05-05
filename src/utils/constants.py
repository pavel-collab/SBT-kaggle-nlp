import torch

model_list = [
    # "bert-base-uncased",
    # "distilbert-base-uncased",
    # "roberta-base",
    # "albert-base-v2",
    # "xlnet-base-cased",
    # "google/electra-base-discriminator",
    # "facebook/bart-base",
    # "microsoft/deberta-base",
    "tbs17/MathBERT",
    
]

large_model_list = [
    'deepseek-ai/deepseek-math-7b-base',
    # 'FacebookAI/roberta-large',
    
    #========================================#
    
    # 'bert-large-uncased',
    # 'deberta-v3-large',
    # 'xlnet-large-cased',
    # 'electra-large-discriminator',
    # 'albert-xxlarge-v2',
    # 'facebook/bart-large',
    # 'google/bigbirf-roberta-large',
    # 'microsoft/mpnet-base',
    # 'sentence-mpnet-base-v2',
]

classes_list = ["Algebra", "Geometry and Trigonometry", "Calculus and Analysis",
                "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math",
                "Linear Algebra", "Abstract Algebra and Topology"]
n_classes = len(classes_list)

train_csv_file = './data/train.csv'
test_csv_file = './data/test.csv'
generated_csv_file = './data/generated/generated_train.csv'

batch_size = 2
num_epoches = 3
num_folds = 5

class ClassWeights:
    class_weights=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

clw = ClassWeights()