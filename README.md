# Sanskrit RAG (CPU-only) - PoC

This repository contains a minimal Retrieval-Augmented Generation (RAG) proof-of-concept for Sanskrit documents using:

- Document ingestion from `data/` (supports `.txt`, `.pdf`, `.docx`)
- Retrieval via TF-IDF char n-grams (works reasonably for Sanskrit/IAST/Unicode)
- Generation via a local GPT-2 model in `model/` (CPU-only)

## Demo

[[Demonstration Video](Demonstration%20Video.mp4)](https://github.com/user-attachments/assets/c4d9cbf6-00b9-4a06-9945-ea792491fe1c)

## Setup (Windows)

Use the provided `venv/` (already present in this repo).

```bat
venv\Scripts\python -m pip install -r requirements.txt
```

## 1) Ingest documents

```bat
venv\Scripts\python code\ingest.py --data_dir data --out artifacts\chunks.jsonl
```

## 2) Build index

```bat
venv\Scripts\python code\index.py --chunks artifacts\chunks.jsonl --out artifacts\tfidf_index.joblib
```

## 3) Ask questions (RAG)

```bat
venv\Scripts\python code\rag_cli.py --query "कालीदासः किम् अकरोत् ?" --top_k 4 --max_new_tokens 120
```

By default, generation uses a CPU instruction-tuned seq2seq model (mt5-small) for better QA-style answers:

```bat
venv\Scripts\python code\rag_cli.py --backend flan_t5 --gen_model google/mt5-small --query "कालीदासः किम् अकरोत् ?" --top_k 3 --max_new_tokens 80
```

### Input formats (Sanskrit + transliteration)

You can ask in:

- Sanskrit (Devanagari), e.g. `धर्मः किम्?`
- Simple English letters (transliteration), e.g. `dharma kya hai`

The CLI will try to transliterate latin input to Devanagari using `indic_transliteration`.

### Answer modes

By default, the system returns an **extractive Sanskrit sentence** from the retrieved context (more stable output).

- Extractive (default):

```bat
venv\Scripts\python code\rag_cli.py --answer_mode extractive --query "शीतं बहु बाधति कः उक्तवान् ?" --top_k 3
```

- Generative (CPU model):

```bat
venv\Scripts\python code\rag_cli.py --answer_mode generate --backend flan_t5 --gen_model google/mt5-small --query "शीतं बहु बाधति कः उक्तवान् ?" --top_k 3 --max_new_tokens 80
```

If you want to use the local GPT-2 model in `model/` instead:

```bat
venv\Scripts\python code\rag_cli.py --backend gpt2 --model_dir model --query "कालीदासः किम् अकरोत् ?" --top_k 3 --max_new_tokens 80
```

You can also query in transliteration/English if your documents contain that text.

### Windows note: non-ASCII queries

If you see garbled Sanskrit when passing `--query "..."`, use `--query_file`.

PowerShell recommended (UTF-8):

```powershell
Set-Content -Path artifacts\q2.txt -Value "शीतं बहु बाधति कः उक्तवान् ?" -Encoding utf8
venv\Scripts\python code\rag_cli.py --backend flan_t5 --gen_model google/mt5-small --query_file artifacts\q2.txt --top_k 3 --max_new_tokens 80
```

Note: If you created the file via `echo ... > file`, it may be UTF-16; the script also supports UTF-16 BOM.

## Notes

- This PoC focuses on being runnable and CPU-only.
- Put `.pdf` files into `data/` for PDF ingestion.
- The first time you run FLAN-T5, HuggingFace will download the model to your local cache (internet required once).
- If you change retrieval settings (vectorizer), rebuild the index by rerunning the index step.
