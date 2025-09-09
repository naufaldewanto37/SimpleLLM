def format_example(example):
    sys = "You are a helpful bilingual (ID+EN) assistant."
    inst = example["instruction"].strip()
    inp  = example.get("input","").strip()
    tgt  = example["output"].strip()
    if inp:
        user = f"Instruction: {inst}\nInput: {inp}"
    else:
        user = f"Instruction: {inst}"

    prompt = f"<|system|>\n{sys}\n<|user|>\n{user}\n<|assistant|>\n{tgt}"
    return {"text": prompt}