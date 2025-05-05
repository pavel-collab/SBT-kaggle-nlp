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

#TODO: последние 2 датасеты очень похожи, может быть их можно объединить?      
class MathTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, important_words_dict, label2str):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.important_words_dict = important_words_dict
        self.label2str = label2str
        
        # Все ключевые слова, используемые для создания фичей
        self.all_hint_words = sorted(set().union(*(important_words_dict[label] for label in label2str)))
        self.num_hint_features = len(self.all_hint_words)

    def extract_hint_features(self, text, label):
        tokens = self.tokenizer.tokenize(text.lower())
        return [1 if any(w in t for t in tokens) else 0 for w in self.all_hint_words]

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        hints = self.extract_hint_features(text, label)
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'hint_features': torch.tensor(hints, dtype=torch.float),
            'labels': torch.tensor(label)
        }

    def __len__(self):
        return len(self.texts)
    
from torch.utils.data import Dataset
import torch

class InferenceDatasetWithHints(Dataset):
    def __init__(self, texts, tokenizer, important_words_dict, label2str_pred):
        """
        texts — список текстов для классификации
        tokenizer — токенизатор (например, от BERT)
        important_words_dict — словарь: класс -> множество ключевых слов
        label2str_pred — список классов, по которым составляется union ключевых слов
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.important_words_dict = important_words_dict
        self.label2str_pred = label2str_pred

        # Все ключевые слова, используемые для создания фичей
        self.all_hint_words = sorted(set().union(*(important_words_dict[label] for label in label2str_pred)))
        self.num_hint_features = len(self.all_hint_words)

    def extract_hint_features(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        return [1 if any(word in token for token in tokens) else 0 for word in self.all_hint_words]

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        hint_features = self.extract_hint_features(text)

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'hint_features': torch.tensor(hint_features, dtype=torch.float)
        }

    def __len__(self):
        return len(self.texts)
    
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
    
class InferenceDatasetWithAttentionHints(Dataset):
    def __init__(self, texts, tokenizer, hint_vocab, max_length=128):
        """
        texts       — список текстов для классификации
        tokenizer   — токенизатор (например, от BERT)
        hint_vocab  — список всех ключевых слов, общий с обучением
        max_length  — максимальная длина входной последовательности
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.hint_vocab = hint_vocab
        self.max_length = max_length

    def extract_hint_features(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        return [1 if any(hint in token for token in tokens) else 0 for hint in self.hint_vocab]

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        hint_features = self.extract_hint_features(text)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),         # [seq_len]
            'attention_mask': encoded['attention_mask'].squeeze(0),  # [seq_len]
            'hint_features': torch.tensor(hint_features, dtype=torch.float)  # [len(hint_vocab)]
        }

    def __len__(self):
        return len(self.texts)