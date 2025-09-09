from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import math
from typing import List, Optional
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
from io import BytesIO

try:
    from pypdf import PdfReader
except:
    PdfReader = None
try:
    import docx
except:
    docx = None

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

SYS_SUMMARY = (
    "You are a helpful bilingual (ID+EN) assistant. "
    "Summarize clearly in Indonesian unless asked otherwise. "
    "Keep key facts, entities, numbers; remove fluff."
)

def chat_generate(system: str, user: str, max_new_tokens=256, temperature=0.3, top_p=0.9):
    messages = [
        {"role":"system","content":system},
        {"role":"user","content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
        )
    new_tokens = out.sequences[0][ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def model_ctx_len() -> int:
    m = getattr(tok, "model_max_length", 4096)
    try:
        return 4096 if m is None or m > 10_000_000 else int(m)
    except:
        return 4096

def smart_chunks(text: str, target_tokens: int = 1024) -> List[str]:
    approx_chars = target_tokens * 4
    parts = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + approx_chars)
        cut = text.rfind("\n", start, end)
        if cut == -1: cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start: cut = end
        parts.append(text[start:cut].strip())
        start = cut
    return [p for p in parts if p]

def summarize_text(text: str, bullets: bool = True, max_new_tokens: int = 256) -> str:
    ctx = model_ctx_len()
    chunk_tokens = max(256, min(1024, ctx // 3))
    chunks = smart_chunks(text, target_tokens=chunk_tokens)

    partial = []
    for i, c in enumerate(chunks, 1):
        instr = (
            "Ringkas bagian dokumen berikut. Fokuskan pada poin penting, entitas, angka, dan kesimpulan. "
            "Hasilkan ringkasan singkat (5–8 bullet) jika memungkinkan.\n\n"
            f"=== BAGIAN {i}/{len(chunks)} ===\n{c}"
        )
        summary = chat_generate(SYS_SUMMARY, instr, max_new_tokens=max_new_tokens)
        partial.append(summary)

    join = "\n\n".join(f"- {s}" for s in partial)
    final_instr = (
        "Gabungkan ringkasan parsial menjadi satu ringkasan komprehensif, terstruktur, dan padat.\n"
        "Sertakan: tujuan dokumen, poin utama, metrik/angka penting, dan rekomendasi/aksi (bila ada)."
        + ("\nOutput dalam bullet points rapi." if bullets else "")
        + "\n\nRINGKASAN PARSIAL:\n" + join
    )
    return chat_generate(SYS_SUMMARY, final_instr, max_new_tokens=max_new_tokens)

def read_pdf(file: UploadFile) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf belum terpasang. pip install pypdf")
    data = BytesIO(file.file.read())
    pdf = PdfReader(data)
    texts = []
    for page in pdf.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def read_docx(file: UploadFile) -> str:
    if docx is None:
        raise RuntimeError("python-docx belum terpasang. pip install python-docx")
    data = BytesIO(file.file.read())
    d = docx.Document(data)
    return "\n".join(p.text for p in d.paragraphs)

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

class SummReq(BaseModel):
    text: str
    bullets: bool = True
    max_new_tokens: int = 256

@app.get("/")
def root():
    return {"ok": True, "msg": "LLM API up. See /docs"}

@app.get("/health")
def health():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": dev}


@app.post("/summarize")
def summarize(req: SummReq):
    if not req.text or not req.text.strip():
        return {"summary": ""}
    summary = summarize_text(req.text, bullets=req.bullets, max_new_tokens=req.max_new_tokens)
    return {"summary": summary}

@app.post("/summarize_file")
def summarize_file(
    file: UploadFile = File(...),
    bullets: bool = Form(True),
    max_new_tokens: int = Form(256),
):
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        text = read_pdf(file)
    elif name.endswith(".docx"):
        text = read_docx(file)
    else:
        text = file.file.read().decode("utf-8", errors="ignore")
    summary = summarize_text(text, bullets=bullets, max_new_tokens=max_new_tokens)
    return {"summary": summary, "chars": len(text)}

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
