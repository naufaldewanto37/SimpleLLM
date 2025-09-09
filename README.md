---
title: Qwen2.5-0.5B + QLoRA (Gradio UI)
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: space_app.py
pinned: false
---


---

# Qwen2.5-0.5B + QLoRA — Training & Inference (HF Spaces + FastAPI)

End-to-end template to **train a QLoRA adapter** and **serve it**:

* 🧪 Train with TRL/PEFT → saves adapter to `outputs-qwen25-05b-qlora/`
* ☁️ Upload adapter to **Hugging Face Hub**
* 🖥️ Inference via **Hugging Face Spaces (Gradio UI)** or a **FastAPI API**

> Base model: **Qwen/Qwen2.5-0.5B-Instruct**
> Supports **private adapter repos** (HF token), streaming, and Space cache fixes.

---

# Qwen2.5-0.5B + QLoRA — Training & Inference

🚀 Live demo available here: [Hugging Face Space](https://huggingface.co/spaces/naufaldewanto37/Chatbot)


## Suggested Repository Structure

```
.
├─ app.py                    # FastAPI (local/server API inference)
├─ space_app.py              # (optional) Gradio UI for Hugging Face Spaces
├─ requirements.txt          # inference/UI dependencies
├─ requirements-train.txt    # training dependencies
├─ training/
│  ├─ train_qlora.py         # QLoRA training script
│  └─ prompt_format.py       # builds chat 'text' from instruction/input/output
├─ scripts/
│  └─ parsing_dataset.py     # example: convert dataset → JSONL SFT format
├─ data/
│  ├─ train.jsonl            # generated SFT dataset
│  └─ val.jsonl              # generated validation split
├─ .gitignore
└─ README.md
```

> If your filenames differ (e.g., only `app.py`), keep them—just adjust the commands below accordingly.

---

## Requirements

**Inference/UI** (`requirements.txt`)

```
transformers==4.44.2
accelerate==0.34.2
peft>=0.13.2
huggingface-hub>=0.24.5
safetensors>=0.4.3
torch>=2.3.0
gradio>=4.44.0
sentencepiece>=0.2.0
einops>=0.7.0
```

**Training** (`requirements-train.txt`)

```
transformers==4.44.2
accelerate==0.34.2
peft>=0.13.2
trl>=0.9.6
datasets>=2.20.0
bitsandbytes>=0.43.0
safetensors>=0.4.3
huggingface-hub>=0.24.5
```

---

## Quickstart

### 0) Environment

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Prepare Dataset → JSONL

```bash
pip install -r requirements-train.txt
python scripts/parsing_dataset.py
# produces: data/train.jsonl and data/val.jsonl
```

**JSONL format** (one JSON object per line):

```json
{"instruction":"...", "input":"...", "output":"..."}
```

### 2) Build Chat Text (prompt templating)

```bash
python training/prompt_format.py
# adds a 'text' field using your chat template
```

### 3) Train QLoRA

```bash
python training/train_qlora.py
# output adapter: outputs-qwen25-05b-qlora/
```

### 4) Upload Adapter to Hugging Face Hub

```bash
huggingface-cli login
huggingface-cli repo create <username>/qwen25-05b-qlora --type model
git clone https://huggingface.co/<username>/qwen25-05b-qlora
cp -r outputs-qwen25-05b-qlora/* <username>/qwen25-05b-qlora/
cd <username>/qwen25-05b-qlora
git lfs install
git add .
git commit -m "Add QLoRA adapter"
git push
```

> Recommended: keep **adapters** on the Hub (do **not** commit weights to GitHub).

---

## Inference Options

### A) Hugging Face Spaces (Gradio UI)

1. Create a **Space** (SDK: *Gradio*, choose CPU/GPU).
2. Add **Secrets** (Settings → *Variables and secrets*):

   * `ADAPTER_REPO_ID` = `<username>/qwen25-05b-qlora`
   * If the adapter repo is **private**: `HF_TOKEN` = your HF token (**read** scope)
   * Optional: `ADAPTER_REVISION` (default `main`), `MERGE_LORA=1`,
     `SYSTEM_PROMPT`, `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`, `REP_PEN`
3. Push:

   * `space_app.py` (Gradio UI)
   * `requirements.txt`
   * `README.md`
4. If you use this README directly in Spaces, add **front-matter** at the very top:

   ```yaml
   ---
   title: Qwen2.5-0.5B + QLoRA (Gradio UI)
   emoji: 🧠
   colorFrom: indigo
   colorTo: purple
   sdk: gradio
   sdk_version: "4.44.0"
   app_file: space_app.py
   pinned: false
   ---
   ```

**Notes**

* `space_app.py` streams tokens, supports **private adapters** (sends `HF_TOKEN` to Transformers **and** PEFT), and forces cache to `./.hfhome` to avoid `/data` permission errors in Spaces.

### B) Local/Server API (FastAPI)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

> You can make `app.py` load from a **local adapter folder** or directly from **HF Hub** via environment variables, similar to the Space.

---

## Configuration (Environment Variables)

| Name               | Used in   | Default                      | Description                                                         |
| ------------------ | --------- | ---------------------------- | ------------------------------------------------------------------- |
| `BASE_MODEL`       | Space/API | `Qwen/Qwen2.5-0.5B-Instruct` | Base LM                                                             |
| `ADAPTER_REPO_ID`  | Space/API | —                            | HF model repo for adapter (e.g., `user/qwen25-05b-qlora`)           |
| `ADAPTER_REVISION` | Space/API | `main`                       | Branch/revision                                                     |
| `HF_TOKEN`         | Space/API | —                            | **Required** if the adapter repo is private (store as a **Secret**) |
| `MERGE_LORA`       | Space/API | `0`                          | `1` merges LoRA into base on load                                   |
| `SYSTEM_PROMPT`    | Space/API | concise helper               | Assistant style                                                     |
| `MAX_NEW_TOKENS`   | Space/API | `512`                        | Generation cap                                                      |
| `TEMPERATURE`      | Space/API | `0.7`                        | Sampling temperature                                                |
| `TOP_P`            | Space/API | `0.9`                        | Nucleus sampling                                                    |
| `REP_PEN`          | Space/API | `1.1`                        | Repetition penalty                                                  |

**Spaces cache fix** (already in code):
`HF_HOME=.hfhome` and `TRANSFORMERS_CACHE=.hfhome/transformers`

---

## Troubleshooting

* **401 Unauthorized / Repository Not Found / Invalid username or password**
  Private adapter repo but the request was unauthenticated.
  → Ensure `HF_TOKEN` is set as a **Secret** and that your code passes it to **Transformers** (`token=...`) **and** **PEFT** (`use_auth_token=...`).

* **`ValueError: Can't find 'adapter_config.json'`**
  Usually the same auth issue or a wrong `ADAPTER_REPO_ID`/`ADAPTER_REVISION`.

* **`TypeError: LoraConfig.__init__() got an unexpected keyword argument 'corda_config'`**
  Your `adapter_config.json` contains extra keys.
  → Upgrade **PEFT to ≥ 0.13.2**, or sanitize the config before loading (see UI code variant).

* **`PermissionError: [Errno 13] /data` (Spaces)**
  Writing cache to a read-only path.
  → Use local cache `./.hfhome` (already handled).

* **Stuck lock file in cache**
  Delete `.hfhome` in the Space Files tab and **Restart**.

---

## Security & Best Practices

* **Do not** commit adapters (`*.safetensors`) or tokens to GitHub. Keep adapters on **HF Hub** and tokens as **Secrets**.
* Use `.gitignore` to exclude `data/`, caches, and large artifacts.
* For public demos, consider request size limits and basic abuse protection.

---

## License

Add a license that fits your project (e.g., MIT/Apache-2.0).

---

## Credits

* [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
* [PEFT](https://github.com/huggingface/peft) · [TRL](https://github.com/huggingface/trl) · [Gradio](https://gradio.app/)

---



