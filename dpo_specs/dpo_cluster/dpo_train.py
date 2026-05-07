import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import random

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)

print("Loading GSM8K dataset...")
dataset = load_dataset("gsm8k", "main")

def create_synthetic_dpo_pairs(dataset_split, num_samples=1000):
    dpo_data = {"prompt": [], "chosen": [], "rejected": []}
    samples = list(dataset_split)[:num_samples]
    for i, item in enumerate(samples):
        prompt = f"Question: {item['question']}\nAnswer:"
        correct_completion = item['answer']
        random_incorrect_item = random.choice([x for j, x in enumerate(samples) if j != i])
        incorrect_completion = random_incorrect_item['answer']
        dpo_data["prompt"].append(prompt)
        dpo_data["chosen"].append(correct_completion)
        dpo_data["rejected"].append(incorrect_completion)
    return Dataset.from_dict(dpo_data)

print("Constructing preference pairs...")
train_dpo_dataset = create_synthetic_dpo_pairs(dataset['train'], num_samples=2000)
eval_dpo_dataset  = create_synthetic_dpo_pairs(dataset['test'],  num_samples=200)

training_args = TrainingArguments(
    output_dir="/scratch/rarora8/madhu/dpo_experiment/output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="none"
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=train_dpo_dataset,
    eval_dataset=eval_dpo_dataset,
    tokenizer=tokenizer,
    max_prompt_length=256,
    max_length=512,
)

print("Starting DPO Training...")
dpo_trainer.train()
dpo_trainer.save_model("/scratch/rarora8/madhu/dpo_experiment/final_model")
print("Done.")