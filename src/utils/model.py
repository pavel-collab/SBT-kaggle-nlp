from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
from utils.focal_loss import FocalLoss

class BertWithFocalLoss(BertPreTrainedModel):
    def __init__(self, config, focal_loss=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.focal_loss = focal_loss or FocalLoss()

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs[1]  # CLS токен
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)

        return {"loss": loss, "logits": logits}