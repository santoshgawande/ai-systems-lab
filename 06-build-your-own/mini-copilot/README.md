# mini-copilot

A terminal-based code completion assistant. Type code, get AI completions. Uses Ollama locally — no API key.

## What it teaches

- Code context injection: how to assemble a useful prompt from surrounding code
- Prefix-suffix (PSM) prompting: give the model code BEFORE and AFTER cursor
- Fill-in-the-middle (FIM): how GitHub Copilot and Codex actually work
- Streaming completions: render tokens as they arrive

## Run

```bash
python copilot.py
```

## How it works

```
[existing code before cursor]
▼
System: "Complete the code. Only output the completion, nothing else."
Prompt: "PREFIX:\n{code_before}\n\nSUFFIX:\n{code_after}\n\nCOMPLETION:"
▼
[model generates completion]
▼
[inserted between prefix and suffix]
```

## Usage modes

1. **Completion mode**: provide code before/after, get the middle filled in
2. **Function generation**: provide function signature + docstring, get implementation
3. **Test generation**: provide function, get test cases

## FIM vs standard completion

Standard:
```
Complete: def add(a, b):
→ return a + b
```

FIM (fill-in-middle):
```
PREFIX: def calculate(items):
    total = 0
    <FILL>
SUFFIX:
    return total
→ for item in items:
       total += item.price
```

FIM is how Copilot works — it knows both what came before AND what should come after.
