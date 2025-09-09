import os, threading, torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

# ---------- Konfigurasi ----------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_REPO_ID = os.environ.get("ADAPTER_REPO_ID", "")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a concise, helpful assistant.")
MERGE_LORA = os.environ.get("MERGE_LORA", "0") == "1"

HF_HOME = os.path.join(os.getcwd(), ".hfhome")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------- Load Tokenizer & Model ----------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, 
    use_fast=True, 
    trust_remote_code=True, 
    padding_side="left", 
    cache_dir=os.environ["TRANSFORMERS_CACHE"],
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=os.environ["TRANSFORMERS_CACHE"],
)

model = PeftModel.from_pretrained(
    base,
    ADAPTER_REPO_ID,
    torch_dtype=dtype,
    cache_dir=os.environ["TRANSFORMERS_CACHE"],
)

if MERGE_LORA:
    model = model.merge_and_unload()

model.eval()

# ---------- Chat logic (streaming) ----------
def chat_generate(message, history):
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # Gunakan template chat resmi dari tokenizer
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# ---------- UI ----------
demo = gr.ChatInterface(
    fn=chat_generate,
    title="Qwen2.5-0.5B + QLoRA (HF Space)",
    description=(
        "Inference QLoRA adapter di atas base **Qwen/Qwen2.5-0.5B-Instruct**.\n"
        "- Set **ADAPTER_REPO_ID** di Space secrets ke repo LoRA Anda (contoh: `username/qwen25-05b-qlora`).\n"
        "- Opsi: set **MERGE_LORA=1** untuk merge LoRA saat load.\n"
        "- Model ini sudah streaming token-by-token."
    ),
)

if __name__ == "__main__":
    demo.launch()
