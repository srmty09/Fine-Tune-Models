import torch
from transformers import AutoTokenizer, TextStreamer
from peft import AutoPeftModelForCausalLM
from colorama import Fore, Style, init

init(autoreset=True)

REPO_NAME = "srmty/smollm2-python-coder"

print(f"{Fore.CYAN}Loading model from {REPO_NAME}{Style.RESET_ALL}")

tokenizer = AutoTokenizer.from_pretrained(REPO_NAME)

model = AutoPeftModelForCausalLM.from_pretrained(
    REPO_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

print(f"\n{Fore.GREEN} Model loaded successfully!{Style.RESET_ALL}\n")

print(f"{Fore.MAGENTA}{'=' * 60}")
print("           SmolLM2 Python Coder")
print(f"{'=' * 60}{Style.RESET_ALL}")

print(f"{Fore.YELLOW}Base Model :{Style.RESET_ALL} SmolLM2-350M")
print(f"{Fore.YELLOW}Parameters :{Style.RESET_ALL} 350 Million")
print(f"{Fore.YELLOW}Fine-Tuning:{Style.RESET_ALL} Supervised Fine-Tuning (SFT)")
print(f"{Fore.YELLOW}Domain     :{Style.RESET_ALL} Python Code Generation")
print(f"{Fore.YELLOW}Adapter    :{Style.RESET_ALL} LoRA (Low-Rank Adaptation)")
print(f"{Fore.YELLOW}Repository :{Style.RESET_ALL} {REPO_NAME}")

print(f"\n{Fore.CYAN}Type 'quit', 'exit', or 'q' to exit.{Style.RESET_ALL}\n")

SYSTEM_PROMPT = (
    "You are a highly skilled and efficient Python code generator. "
    "Provide clear, correct, and optimized Python code based on the user's request."
)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

def generate(instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

while True:
    try:
        print()
        instruction = input(
            f"{Fore.GREEN}user{Style.RESET_ALL}: "
        ).strip()

        if instruction.lower() in ("quit", "exit", "q"):
            print(f"\n{Fore.RED}bye!{Style.RESET_ALL}")
            break

        if not instruction:
            continue

        print(f"\n{Fore.BLUE}assistant{Style.RESET_ALL}: ")
        generate(instruction)
        print()

    except KeyboardInterrupt:
        print(f"\n{Fore.RED}bye!{Style.RESET_ALL}")
        break