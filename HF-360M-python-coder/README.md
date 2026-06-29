---

base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
tags:

* smollm2
* python
* code-generation
* sft
  language:
* en

---

# smollm2-python-coder

A small Python coding model fine-tuned from `HuggingFaceTB/SmolLM2-360M-Instruct`.

## Model Details

* **Base model:** `HuggingFaceTB/SmolLM2-360M-Instruct`
* **Model type:** Causal Language Model
* **Task:** Python code generation
* **Language:** English
* **Fine-tuning method:** Supervised Fine-Tuning

## Dataset

This model was fine-tuned using Python coding instruction data.

## Usage

```python
import torch
from transformers import pipeline

model_id = "srmty/smollm2-python-coder"

generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

messages = [
    {
        "role": "user",
        "content": "Write a Python function to check whether a number is prime."
    }
]

output = generator(
    messages,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    return_full_text=False,
    pad_token_id=generator.tokenizer.eos_token_id,
)

print(output[0]["generated_text"])
```

## Intended Use

This model is intended for simple Python coding tasks, including:

* Writing Python functions
* Explaining Python code
* Fixing simple bugs
* Solving beginner-level programming problems

## Training

This model was trained using supervised fine-tuning.
