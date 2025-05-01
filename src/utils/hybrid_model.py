from transformers import BertModel
import torch.nn as nn
import torch
from safetensors.torch import load_file

class HybridModelHF(nn.Module):
    def __init__(self, num_labels, extra_feat_dim, model_path_str=None):
        super().__init__()
        # if model_path_str is not None:
        #     self.bert = BertModel.from_pretrained(model_path_str)
        # else:
        #     self.bert = BertModel.from_pretrained('bert-base-uncased')
            
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + extra_feat_dim, num_labels)

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        concat = torch.cat((pooled, extra_features), dim=1)
        logits = self.classifier(self.dropout(concat))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits}