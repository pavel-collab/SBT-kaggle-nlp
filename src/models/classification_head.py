import torch.nn as nn
    
class ClassificationHead2(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead2, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)
    
class ClassificationHead5(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead5, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)