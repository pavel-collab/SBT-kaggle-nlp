import torch

model_list = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "xlnet-base-cased",
    "google/electra-base-discriminator",
    "facebook/bart-base",
    "microsoft/deberta-base"
]

classes_list = ["Algebra", "Geometry and Trigonometry", "Calculus and Analysis",
                "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math",
                "Linear Algebra", "Abstract Algebra and Topology"]
n_classes = len(classes_list)

train_csv_file = './data/train.csv'
test_csv_file = './data/test.csv'
generated_csv_file = './data/generated/generated_train.csv'

batch_size = 8
num_epoches = 5

class_weights=torch.tensor([0.4903152069297401, 
                            0.5230364476386037, 
                            1.2290410132689988, 
                            3.4655612244897958, 
                            0.7324766355140186, 
                            0.6964285714285714, 
                            13.40625, 
                            14.151041666666666])