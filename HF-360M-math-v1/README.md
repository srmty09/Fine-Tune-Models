# smolLM2-360M-instruct-math-v1

A small instruction and math-tuned language model based on `HuggingFaceTB/SmolLM2-360M`.

## Model Details

* **Base model:** `HuggingFaceTB/SmolLM2-360M`
* **Model type:** Causal Language Model
* **Task:** Instruction following and math reasoning
* **Language:** English
* **Fine-tuning method:** Supervised Fine-Tuning

## Dataset

This model was fine-tuned using a mix of general instruction data and math instruction data.

```text
teknium/GPTeacher-General-Instruct
Math reasoning dataset
```

The GPTeacher dataset was used for general instruction-following, while the math dataset was added to improve step-by-step math reasoning.

## Prompt Format

The model follows an Alpaca-style instruction format:

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```
