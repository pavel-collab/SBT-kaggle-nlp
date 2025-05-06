import torch
from torch import nn
from unsloth import FastLanguageModel
from models.classification_head import *

class UnslothCustomClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        num_labels: int = 2,
        max_seq_length: int = 512,
        quantized: bool = True,
        dtype = None  # можно указать torch.float16 / torch.bfloat16 для ускорения на GPU
    ):
        super().__init__()

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=quantized
        )

        self.hidden_size = self.model.config.hidden_size
        self.classifier = ClassificationHead2(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state
        cls_rep = last_hidden[:, 0, :]
        logits = self.classifier(cls_rep)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
