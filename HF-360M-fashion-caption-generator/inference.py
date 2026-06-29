import argparse
import yaml
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


HF_REPO_ID = "srmty/HF-360M-fashion-caption-generator"
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
    def __init__(self, repo_id):
        super().__init__()

        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.clip = CLIPVisionModel.from_pretrained(self.cfg["clip_id"])
        self.projector = Projector(self.cfg["clip_hidden"], self.cfg["llm_hidden"])

        projector_path = hf_hub_download(repo_id=repo_id, filename="projector.pt")
        self.projector.load_state_dict(torch.load(projector_path, map_location=device))

        dtype = torch.float16 if device.type == "cuda" else torch.float32

        base_llm = AutoModelForCausalLM.from_pretrained(
            self.cfg["llm_id"],
            torch_dtype=dtype,
        )

        self.llm = PeftModel.from_pretrained(base_llm, repo_id, subfolder="lora_adapter")
        self.image_processor = CLIPImageProcessor.from_pretrained(self.cfg["clip_id"])
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def word_embed(self):
        llm = self.llm
        if hasattr(llm, "base_model"):
            llm = llm.base_model
        if hasattr(llm, "model"):
            llm = llm.model
        if hasattr(llm, "model"):
            llm = llm.model
        return llm.embed_tokens

    @torch.no_grad()
    def caption(self, image_path):
        self.eval()

        image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(device)

        clip_dtype = next(self.clip.parameters()).dtype
        image_features = self.clip(pixel_values=pixel_values.to(clip_dtype)).last_hidden_state.float()

        image_embeds = self.projector(image_features)

        messages = [
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user", "content": self.cfg["user_prompt"]},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        text_embeds = self.word_embed()(enc["input_ids"])

        image_embeds = image_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_mask = torch.ones(
            1,
            image_embeds.size(1),
            dtype=enc["attention_mask"].dtype,
            device=device,
        )

        attention_mask = torch.cat([image_mask, enc["attention_mask"]], dim=1)

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg["max_new_tokens"],
            do_sample=False,
            num_beams=3,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if "\n" in text:
            text = text.split("\n")[0].strip()

        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--repo_id", default=HF_REPO_ID)
    args = parser.parse_args()

    model = FashionCaptionModel(args.repo_id).to(device)
    print(model.caption(args.image))


if __name__ == "__main__":
    main()
