import torch
from torch.utils.data import Dataset

class MathTextWithAttentionHints(Dataset):
    def __init__(self, texts, labels, tokenizer, important_words_dict, label2str):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.important_words_dict = important_words_dict
        self.label2str = label2str

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        important = self.important_words_dict[self.label2str[label]]
        hint_mask = torch.tensor([1 if any(w in t.lower() for w in important) else 0 for t in tokens])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'hint_mask': hint_mask,
            'labels': torch.tensor(label)
        }

    def __len__(self):
        return len(self.texts)

class DataCollatorWithHintMask:
    def __call__(self, features):
        return {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'hint_mask': torch.stack([f['hint_mask'] for f in features]),
            'labels': torch.tensor([f['labels'] for f in features])
        }