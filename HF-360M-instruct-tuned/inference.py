import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "srmty/smolLM_360M_Base_it"

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="final")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder="final",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

def make_prompt(instruction, input_text=""):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

def generate(instruction, input_text="", max_new_tokens=256, temperature=0.7, top_p=0.9):
    prompt = make_prompt(instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()

print(generate("Explain what machine learning is in simple words."))