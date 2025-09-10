import os, threading, torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
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

# ---------- Configuration ----------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_REPO_ID = os.environ.get("ADAPTER_REPO_ID", "qwen25-05b-qlora")
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
def chat_generate_to_str(message, history=None, max_new_tokens=512):
    history = history or []
    last = ""
    for partial in chat_generate(message, history, max_new_tokens=max_new_tokens):
        last = partial
    return last.strip()

def chat_generate(message, history, max_new_tokens):
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

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
        max_new_tokens= max_new_tokens,
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

# ---------- Summarization logic ----------
def model_ctx_len() -> int:
    m = getattr(tokenizer, "model_max_length", 4096)
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
            "Summary this document. Focust on key points, entities, numbers, and conclusions. "
            "Give Output 5-8 Point \n\n"
            f"=== Section {i}/{len(chunks)} ===\n{c}"
        )
        summary = chat_generate_to_str(instr, history=[], max_new_tokens=max_new_tokens)
        partial.append(summary)

    join = "\n\n".join(f"- {s}" for s in partial)
    final_instr = (
        "Assemble a comprehensive, structured, and concise summary from the partial summaries below.\n "
        "Give, what is the most important information from the document.\n"
        + ("\nOutput at points" if bullets else "")
        + "\n\nPartial Summary:\n" + join
    )
    final_summary = chat_generate_to_str(final_instr, history=[], max_new_tokens=max_new_tokens)
    return final_summary

def read_pdf(file: UploadFile) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. pip install pypdf")
    with open(file, "rb") as f:
        pdf = PdfReader(f)
        texts = [(page.extract_text() or "") for page in pdf.pages]
    return "\n".join(texts)

def read_docx(file: UploadFile) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed. pip install python-docx")
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)

def summarize_file(filepath, bullets=True, max_new_tokens=256):
    if not filepath:
        return "Please upload a file."
    low = filepath.lower()
    if low.endswith(".pdf"):
        content = read_pdf(filepath)
    elif low.endswith(".docx"):
        content = read_docx(filepath)
    elif low.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    else:
        return "Unsupported file type. Use .pdf, .docx, or .txt"

    if not content or content.startswith("[Error]"):
        return content or "No readable content."
    return summarize_text(content, bullets=bullets,  max_new_tokens=max_new_tokens)

# ---------- UI ----------
with gr.Blocks(title="LLM Toolkit: Chat & Summaries", fill_height=True) as demo:
    gr.Markdown("## 🔧 LLM Toolkit — Chat, Summary Text, Summary File")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            sl_max_new = gr.Slider(32, 2048, value=512, step=32, label="Max New Tokens (Chat)")

            gr.Markdown("#### Summarization")
            sl_max_new_sum = gr.Slider(32, 1024, value=256, step=32, label="Max New Tokens (Summary)")
            ck_bullets = gr.Checkbox(value=True, label="Bullet points output")

        with gr.Column(scale=3):
            with gr.Tabs():
                # ---- Chat Tab ----
                with gr.Tab("💬 Chat"):
                    chat = gr.Chatbot(height=420, show_copy_button=True)
                    chat_state = gr.State([])
                    chat_input = gr.Textbox(placeholder="Type your prompt…", label="Your message", lines=2)
                    with gr.Row():
                        btn_send = gr.Button("Send", variant="primary")
                        btn_clear = gr.Button("Clear")

                    def _on_send(msg, history, m):
                        stream = chat_generate(msg, history or [],  max_new_tokens=m)
                        partial = ""
                        for chunk in stream:
                            partial = chunk
                            yield history + [(msg, partial)]
                        # finalize
                        yield history + [(msg, partial)]

                    btn_send.click(
                        _on_send,
                        inputs=[chat_input, chat_state, sl_max_new],
                        outputs=chat
                    ).then(lambda h: h, chat, chat_state).then(lambda: "", None, chat_input)

                    btn_clear.click(lambda: ([], []), None, [chat, chat_state])

                # ---- Summary Text Tab ----
                with gr.Tab("📝 Summary Text"):
                    txt_input = gr.Textbox(lines=14, label="Paste text here")
                    btn_sum_text = gr.Button("Summarize", variant="primary")
                    txt_out = gr.Markdown()

                    btn_sum_text.click(
                        summarize_text,
                        inputs=[txt_input, ck_bullets,  sl_max_new_sum],
                        outputs=txt_out
                    )

                # ---- Summary File Tab ----
                with gr.Tab("📄 Summary File"):
                    file_input = gr.File(
                        label="Upload .pdf / .docx / .txt",
                        file_types=[".pdf", ".docx", ".txt"],
                        type="filepath"
                    )
                    btn_sum_file = gr.Button("Summarize File", variant="primary")
                    file_out = gr.Markdown()

                    btn_sum_file.click(
                        summarize_file,
                        inputs=[file_input, ck_bullets, sl_max_new_sum],
                        outputs=file_out
                    )

if __name__ == "__main__":
    demo.launch()
