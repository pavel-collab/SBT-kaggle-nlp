from transformers import BertModel, PreTrainedModel
import torch.nn as nn
import torch
from safetensors.torch import load_file

class HybridModelHF(nn.Module):
    def __init__(self, num_labels, extra_feat_dim, model_path_str=None):
        super().__init__()            
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
    
class BertWithHints(PreTrainedModel):
    def __init__(self, config, num_hint_features):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.hint_fc = nn.Linear(num_hint_features, 32)
        self.classifier = nn.Linear(config.hidden_size + 32, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, hint_features=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        hint_embeds = torch.relu(self.hint_fc(hint_features))
        combined = torch.cat([pooled_output, hint_embeds], dim=1)
        logits = self.classifier(combined)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}