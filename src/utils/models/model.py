import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomRobertaClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-large",
        num_labels: int = 8,
        use_lora: bool = False
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        compute_dtype = getattr(torch, "float16")

        # BitsAndBytes config
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load backbone with quantization
        self.backbone = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Apply LoRA (optional)
        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            
        self.backbone.config.use_cache = False
        self.backbone.config.pretraining_tp = 1
        
        #! Freezing backbone model parameters
        # #freezing backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Classification head
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        ).to(torch.float16) #! если не привести к нужной точности, при прямом прогоне backbone и голова будут иметь разные типы

    def forward(self, 
                input_ids, 
                attention_mask=None, 
                # token_type_ids=None,
                position_ids=None, 
                # head_mask=None, 
                labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]  # [CLS] токен
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None
        )

    def tokenize(self, texts, max_length=512):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )