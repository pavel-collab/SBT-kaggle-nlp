import torch

model_list = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    # "roberta-base",
    # "albert-base-v2",
    # "xlnet-base-cased",
    # "google/electra-base-discriminator",
    "facebook/bart-base",
    # "microsoft/deberta-base"
]

classes_list = ["Algebra", "Geometry and Trigonometry", "Calculus and Analysis",
                "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math",
                "Linear Algebra", "Abstract Algebra and Topology"]
n_classes = len(classes_list)

train_csv_file = './data/train.csv'
test_csv_file = './data/test.csv'
generated_csv_file = './data/generated/generated_train.csv'

batch_size = 8
num_epoches = 3

class ClassWeights:
    class_weights=torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

clw = ClassWeights()