import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model


# -----------------------------
# Config
# -----------------------------
DATASET_ID = "jtatman/python-code-dataset-500k"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

OUTPUT_DIR = "./smollm2-python-coder"

UNIFIED_SYSTEM_PROMPT = (
    "You are a highly skilled and efficient Python code generator. "
    "Provide clear, correct, and optimized Python code based on the user's request."
)


# -----------------------------
# Dataset
# -----------------------------
dataset = load_dataset(DATASET_ID)
train_dataset = dataset["train"]


# -----------------------------
# Model & Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


# -----------------------------
# Format dataset
# -----------------------------
def format_prompt(example):
    messages = [
        {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


formatted_dataset = train_dataset.map(
    format_prompt,
    remove_columns=train_dataset.column_names,
)


# -----------------------------
# LoRA
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -----------------------------
# Training Args
# TRL 1.5.x uses max_length instead of max_seq_length
# -----------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,
    bf16=True,
    save_steps=100,
    dataset_text_field="text",
    max_length=1024,
    packing=False,
)


# -----------------------------
# Trainer
# tokenizer is deprecated in newer TRL;
# use processing_class instead.
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    processing_class=tokenizer,
)


# -----------------------------
# Train
# -----------------------------
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)

    print(f"Training complete. Model saved to: {OUTPUT_DIR}")
