import torch
from transformers import Trainer
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Используйте свой оптимизатор
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        # Если нужно, можете добавить свой scheduler здесь
        # Создание планировщика
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,  # Количество шагов для разогрева
            num_training_steps=num_training_steps  # Общее количество шагов обучения
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # Получение выходов модели
        outputs = model(**inputs)
        # Используйте вашу собственную функцию потерь
        # Например, можно использовать стандартную CrossEntropyLoss
        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        # Предполагается, что вам нужны логиты и метки
        logits = outputs.logits
        labels = inputs.get("labels")
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss