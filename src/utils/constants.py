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
    'deepseek-ai/deepseek-math-7b-base', # 7b
    'FacebookAI/roberta-large', # 356M
    # 'google-bert/bert-large-uncased', # 336M
    # 'microsoft/deberta-v3-large',
    # 'xlnet/xlnet-large-cased',
    # 'google/electra-large-discriminator',
    # 'albert/albert-xxlarge-v2', # 223M
    # 'facebook/bart-large',
    # 'google/bigbird-roberta-large', #! maybe will be need special BigBirdModel.from_pretrained
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