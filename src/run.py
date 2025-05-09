import unsloth
import argparse
from utils.constants import *
from utils.utils import (fix_random_seed,
                         get_device,
                         get_dataset,
                         print_device_info)
from transformers import (TrainingArguments,
                          DataCollatorForLanguageModeling)
from unsloth import FastLanguageModel
from typing import List, Union, Any, Dict
from trl import SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--use_generation', action='store_true', help='if we using generation data for train')
parser.add_argument('-m', '--model_name', type=str, default='unsloth/llama-3-8b-bnb-4bit', help='set model name')
args = parser.parse_args()

fix_random_seed()
device = get_device()

train_dataset = get_dataset(use_generation=args.use_generation,
                            get_class_weight_flag=False)

texts = train_dataset['text']
labels = train_dataset['label']

# print_device_info()
model_name = args.model_name

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,  #! или другая 4bit модель
    max_seq_length = 1024,
    dtype = torch.float16,  # можно float32 при необходимости
    load_in_4bit = True,
)

number_token_ids = []
for i in range(0, n_classes+1):
    number_token_ids.append(tokenizer.encode(str(i), add_special_tokens=False)[0])
# keep only the number tokens from lm_head
par = torch.nn.Parameter(model.lm_head.weight[number_token_ids, :])

old_shape = model.lm_head.weight.shape
old_size = old_shape[0]
print(par.shape)
print(old_shape)

model.lm_head.weight = par

reverse_map = {value: idx for idx, value in enumerate(number_token_ids)} # will be used later to convert an idx from the old tokenizer to the new lm_head

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "lm_head", # can easily be trained because it now has a small size
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    # init_lora_weights = 'loftq',
    # loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1), # And LoftQ
)

prompt = """Here is a mathematical problem:
{}

Classify this problem into one of the following:
class 1: Algebra
class 2: Geometry and Trigonometry
class 3: Calculus and Analysis
class 4: Probability and Statistics
class 5: Number Theory
class 6: Combinatorics and Discrete Math
class 7: Linear Algebra
class 8: Abstract Algebra and Topology

SOLUTION
The correct answer is: class {}"""

def formatting_prompts_func(dataset_):
    # this is to fix an issue with the transformers library where the first time this function is called, it is called with a string for some reason
    if isinstance(dataset_['text'], str):
        return [" "]*100
        
    texts = []
    for i in range(len(dataset_['text'])):
        text_ = dataset_['text'][i]
        label_ = dataset_['label'][i] # the csv is setup so that the label column corresponds exactly to the 3 classes defined above in the prompt (important)

        text = prompt.format(text_, label_)

        texts.append(text)
    return texts

class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            # Find the last non-padding token
            last_token_idx = (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
            # Set all labels to ignore_index except for the last token
            batch["labels"][i, :last_token_idx] = self.ignore_index
            # If the last token in the text is, for example, "2", then this was processed with the old tokenizer into number_token_ids[2]
            # But we don't actually want this because number_token_ids[2] could be something like 27, which is now undefined in the new lm_head. So we map it to the new lm_head index.
            batch["labels"][i, last_token_idx] = reverse_map[ batch["labels"][i, last_token_idx].item() ]


        return batch
    
collator = DataCollatorForLastTokenLM(tokenizer=tokenizer)

model.to(device)
    
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    max_seq_length = 1024,
    dataset_num_proc = 1,
    packing = False, # not needed because group_by_length is True
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        learning_rate = 1e-4,
        fp16 = True,
        # bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        num_train_epochs = 1,
        # report_to = "wandb",
        report_to = "none",
        group_by_length = True,
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

#! here must be inference part