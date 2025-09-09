from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "outputs-qwen25-05b-qlora"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, ADAPTER).eval()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.9
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@app.get("/")
def root():
    return {"ok": True, "msg": "LLM API up. See /docs"}

@app.get("/health")
def health():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": dev}

@app.post("/generate")
def generate(req: Req):
    sys = "You are a helpful bilingual (ID+EN) assistant."
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": req.prompt},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=req.max_new_tokens,
            do_sample=True, temperature=0.9, top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
        )

    full = out.sequences[0]
    new_tokens = full[ids["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return {"text": text}
