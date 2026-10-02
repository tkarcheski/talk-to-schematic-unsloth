"""
train_unsloth.py - LoRA fine-tune Qwen3.8-27B (vision) to review and chat about schematics.

Target box: 1x RTX PRO 6000 Blackwell 96 GB.
Unsloth's Qwen3.8 guide: QLoRA fits in 24 GB, 16-bit LoRA needs >36 GB, so on
96 GB we use 16-bit LoRA (better quality than 4-bit QLoRA) with room for
large schematic images.

  pip install --upgrade unsloth "transformers>=5" trl datasets pillow
  python train_unsloth.py --data data --out outputs/hw-specialist-lora
"""
import argparse
import json
import random

from PIL import Image
from unsloth import FastModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="outputs/hw-specialist-lora")
ap.add_argument("--model", default="unsloth/Qwen3.8-27B")          # use ...-unsloth-bnb-4bit + --4bit for QLoRA
ap.add_argument("--4bit", dest="four_bit", action="store_true")
ap.add_argument("--max-seq", type=int, default=8192)                # 1 image + up to 6 turns
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--general-mix", default=None,
                help="optional JSONL of general chat/vision examples (same format) to reduce forgetting")
ap.add_argument("--general-ratio", type=float, default=0.15)
args = ap.parse_args()


# ---------------------------------------------------------------- data
def load(path):
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        msgs = []
        for m in r["messages"]:
            content = []
            for c in m["content"]:
                if c["type"] == "image":
                    content.append({"type": "image", "image": Image.open(c["image"]).convert("RGB")})
                else:
                    content.append(c)
            msgs.append({"role": m["role"], "content": content})
        out.append({"messages": msgs})
    return out

train = load(f"{args.data}/train.jsonl")
val = load(f"{args.data}/val.jsonl")
if args.general_mix:
    gen = load(args.general_mix)
    random.Random(0).shuffle(gen)
    train += gen[: int(len(train) * args.general_ratio)]
random.Random(0).shuffle(train)
# Unsloth docs recommend a plain Python list (not datasets.map) for vision rows
# so Arrow does not try to standardize the PIL images.
print(f"train={len(train)} val={len(val)}")

# ---------------------------------------------------------------- model
model, tokenizer = FastModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_seq,
    load_in_4bit=args.four_bit,
    full_finetuning=False,
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # schematics are an unusual visual domain: train the vision side too
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=args.rank,
    lora_alpha=args.rank,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Sanity check: render one sample through the chat template and confirm the
# <think> block survives. If your template strips assistant reasoning, move it
# to the field your template expects before training.
FastModel.for_training(model)
probe = tokenizer.apply_chat_template(
    [{"role": m["role"], "content": [c for c in m["content"] if c["type"] == "text"]} for m in train[0]["messages"]],
    tokenize=False)
assert "<think>" in probe or "think" not in train[0]["messages"][-1]["content"][0]["text"], \
    "Chat template dropped the reasoning block - check the template."

# ---------------------------------------------------------------- train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(
        model, tokenizer,
        resize="max",                  # keep full resolution: small text on schematics matters
        train_on_responses_only=True,  # loss on every assistant turn, not the questions
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    train_dataset=train,
    eval_dataset=val,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        max_seq_length=args.max_seq,
        remove_unused_columns=False,          # required for vision
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        output_dir=args.out,
        seed=3407,
        report_to="none",
    ),
)
trainer.train()

model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
# Merged 16-bit weights for vLLM / SGLang serving (then quantize to FP8 if wanted):
model.save_pretrained_merged(args.out + "-merged", tokenizer, save_method="merged_16bit")
print("saved", args.out)
