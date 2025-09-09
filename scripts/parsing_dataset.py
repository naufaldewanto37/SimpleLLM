from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k")
def to_sft(ex):
    return {
        "instruction": ex.get("instruction","").strip(),
        "input": ex.get("context","").strip() or "",
        "output": ex.get("response","").strip()
    }

train = ds["train"].map(to_sft, remove_columns=ds["train"].column_names)
train.to_json("data/train.jsonl", lines=True, force_ascii=False)

val = train.select(range(1000))
val.to_json("data/val.jsonl", lines=True, force_ascii=False)