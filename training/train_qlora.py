# train_qlora.py
import os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from prompt_format import format_example

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "qwen25-05b-qlora"

def main():
    # 1) Tokenizer
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2) 4-bit quant (QLoRA)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 
    )

    # 3) Base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16
    )
    # sinkron id token spesial
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id

    # 4) Dataset
    ds_train = load_dataset("json", data_files="data/train.jsonl")["train"]
    ds_val   = load_dataset("json", data_files="data/val.jsonl")["train"]
    ds_train = ds_train.map(format_example, remove_columns=ds_train.column_names)
    ds_val   = ds_val.map(format_example,   remove_columns=ds_val.column_names)

    # 5) LoRA config
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    # 6) Training config
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        dataset_text_field="text",
        packing=False,
        resume_from_checkpoint=True
    )

    # 7) Trainer
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=lora
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
    main()