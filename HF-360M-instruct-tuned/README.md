# smolLM_360M_Base_it

A small instruction-tuned language model based on `HuggingFaceTB/SmolLM2-360M`.

## Model Details

* **Base model:** `HuggingFaceTB/SmolLM2-360M`
* **Model type:** Causal Language Model
* **Task:** Instruction following
* **Language:** English
* **Fine-tuning method:** Supervised Fine-Tuning

## Dataset

This model was fine-tuned using the **GPTeacher General Instruct** dataset by `teknium`.

```text
teknium/GPTeacher-General-Instruct
```

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
sss