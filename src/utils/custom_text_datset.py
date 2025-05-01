from torch.utils.data import Dataset
import torch

class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'extra_features': self.encodings['extra_features'][idx],
            'labels': self.labels[idx]
        }
    
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, top_words):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        self.extra_features = []
        for text in texts:
            tokens = set(text.lower().split())
            feats = [1.0 if word in tokens else 0.0 for word in top_words]
            self.extra_features.append(feats)
        self.extra_features = torch.tensor(self.extra_features, dtype=torch.float)

    def __len__(self):
        return len(self.extra_features)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'extra_features': self.extra_features[idx]
        }