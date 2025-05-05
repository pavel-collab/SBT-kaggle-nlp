import torch 

class DataCollatorWithHints:
    def __call__(self, features):
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'hint_features': torch.stack([f['hint_features'] for f in features]),
            'labels': torch.tensor([f['labels'] for f in features])
        }
        return batch
    
class DataCollatorWithHintMask:
    def __call__(self, features):
        return {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'hint_mask': torch.stack([f['hint_mask'] for f in features]),
            'labels': torch.tensor([f['labels'] for f in features])
        }