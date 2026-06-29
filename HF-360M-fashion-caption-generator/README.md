# HF-360M Fashion Caption Generator

Simple training and inference code.

## Files

```text
config.yaml
train.py
inference.py
requirements.txt
```

## Train

Edit `hf_repo_id` in `config.yaml` if needed.

```bash
pip install -r requirements.txt
huggingface-cli login
python train.py --config config.yaml
```

The training script saves and pushes:

```text
lora_adapter/
projector.pt
config.yaml
tokenizer files
README.md
```

## Inference from Hugging Face

```bash
python inference.py --image path/to/image.jpg
```

By default it loads:

```text
srmty/HF-360M-fashion-caption-generator
```
