import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm.auto import tqdm
from huggingface_hub import create_repo, HfApi


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Projector(nn.Module):
    def __init__(self, clip_hidden, llm_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_hidden, llm_hidden * 2),
            nn.GELU(),
            nn.Linear(llm_hidden * 2, llm_hidden),
        )

    def forward(self, x):
        return self.net(x)


class FashionCaptionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.clip = CLIPVisionModel.from_pretrained(cfg["clip_id"])
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        self.projector = Projector(cfg["clip_hidden"], cfg["llm_hidden"])

        dtype = torch.float16 if device.type == "cuda" else torch.float32

        base_llm = AutoModelForCausalLM.from_pretrained(
            cfg["llm_id"],
            torch_dtype=dtype,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )

        self.llm = get_peft_model(base_llm, lora_config)
        self.llm.config.use_cache = False

    def word_embed(self):
        llm = self.llm
        if hasattr(llm, "base_model"):
            llm = llm.base_model
        if hasattr(llm, "model"):
            llm = llm.model
        if hasattr(llm, "model"):
            llm = llm.model
        return llm.embed_tokens

    def forward(self, pixel_values, input_ids, labels, attention_mask):
        batch_size = pixel_values.size(0)

        with torch.no_grad():
            clip_dtype = next(self.clip.parameters()).dtype
            image_features = self.clip(pixel_values=pixel_values.to(clip_dtype)).last_hidden_state.float()

        image_embeds = self.projector(image_features)
        text_embeds = self.word_embed()(input_ids)
        image_embeds = image_embeds.to(text_embeds.dtype)

        image_len = image_embeds.size(1)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_mask = torch.ones(
            batch_size,
            image_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        full_attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=None,
            use_cache=False,
        )

        logits = outputs.logits

        text_logits = logits[:, image_len:-1, :].contiguous()
        text_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            text_logits.view(-1, text_logits.size(-1)),
            text_labels.view(-1),
            ignore_index=-100,
        )

        return loss


class FashionDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, cfg):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        caption = item["text"]

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        messages = [
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user", "content": self.cfg["user_prompt"]},
            {"role": "assistant", "content": caption},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not full_text.endswith(self.tokenizer.eos_token):
            full_text += self.tokenizer.eos_token

        prompt_messages = [
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user", "content": self.cfg["user_prompt"]},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        if len(full_ids) > self.cfg["max_text_len"]:
            answer_ids = full_ids[prompt_len:]
            keep = self.cfg["max_text_len"] - prompt_len - 1
            full_ids = prompt_ids + answer_ids[:keep] + [self.tokenizer.eos_token_id]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for x in batch:
        ids = x["input_ids"]
        lbl = x["labels"]
        pad_len = max_len - ids.size(0)

        input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(ids.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def save_and_push(model, tokenizer, cfg):
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    model.llm.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    torch.save(model.projector.state_dict(), os.path.join(output_dir, "projector.pt"))
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    readme = f"""---
license: apache-2.0
base_model: {cfg["llm_id"]}
tags:
- image-captioning
- fashion
- vision-language
- lora
---

# HF-360M Fashion Caption Generator

A simple fashion image captioning model trained on `{cfg["dataset_id"]}`.

It uses:

- `{cfg["clip_id"]}` as frozen vision encoder
- projector MLP
- LoRA adapter on `{cfg["llm_id"]}`
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)

    if cfg["push_to_hub"]:
        create_repo(cfg["hf_repo_id"], private=cfg["hf_private"], exist_ok=True)
        api = HfApi()
        api.upload_folder(
            folder_path=output_dir,
            repo_id=cfg["hf_repo_id"],
            repo_type="model",
        )
        print("Pushed to:", cfg["hf_repo_id"])


def train(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_id"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_processor = CLIPImageProcessor.from_pretrained(cfg["clip_id"])

    raw = load_dataset(cfg["dataset_id"], split="train")
    split = raw.train_test_split(test_size=0.1, seed=42)

    train_dataset = FashionDataset(split["train"], tokenizer, image_processor, cfg)
    eval_dataset = FashionDataset(split["test"], tokenizer, image_processor, cfg)

    collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    model = FashionCaptionModel(cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    total_updates = (len(train_loader) // cfg["grad_accum"]) * cfg["epochs"]
    warmup_steps = int(total_updates * cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_eval_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}")

        for step, batch in enumerate(progress, 1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast("cuda", enabled=device.type == "cuda"):
                loss = model(pixel_values, input_ids, labels, attention_mask)
                loss = loss / cfg["grad_accum"]

            scaler.scale(loss).backward()
            total_loss += loss.item() * cfg["grad_accum"]

            if step % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix(
                loss=f"{total_loss / step:.4f}",
                step=global_step,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="eval"):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast("cuda", enabled=device.type == "cuda"):
                    loss = model(pixel_values, input_ids, labels, attention_mask)

                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        print(f"epoch {epoch} eval loss: {eval_loss:.4f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_and_push(model, tokenizer, cfg)

    print("best eval loss:", best_eval_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
