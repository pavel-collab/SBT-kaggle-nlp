import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedModel, BertConfig

class BertWithAttentionHints(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.attn_weights = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, hint_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        token_embeddings = outputs.last_hidden_state  # [B, T, H]

        # Compute attention weights
        raw_scores = self.attn_weights(token_embeddings).squeeze(-1)  # [B, T]

        # Mask out non-hint tokens
        if hint_mask is not None:
            raw_scores = raw_scores.masked_fill(hint_mask == 0, float('-inf'))

        attention_weights = torch.softmax(raw_scores, dim=1)  # [B, T]
        attended = torch.sum(token_embeddings * attention_weights.unsqueeze(-1), dim=1)  # [B, H]

        logits = self.classifier(attended)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits, 'attention': attention_weights}