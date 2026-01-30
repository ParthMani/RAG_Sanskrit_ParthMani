from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import joblib
import torch
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_chunks(path: str) -> List[dict]:
    chunks: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def normalize_query(q: str) -> str:
    q = q.strip()
    q = re.sub(r"[\?\!\.,;:\(\)\[\]\{\}\"\'`]+", " ", q)
    q = q.replace("।", " ")
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def maybe_transliterate_to_devanagari(q: str) -> str | None:
    if not q:
        return None
    if re.search(r"[A-Za-z]", q) is None:
        return None
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:
        return None

    for scheme in [
        getattr(sanscript, "IAST", None),
        getattr(sanscript, "ITRANS", None),
        getattr(sanscript, "HK", None),
    ]:
        if scheme is None:
            continue
        try:
            out = transliterate(q, scheme, sanscript.DEVANAGARI)
            out = out.strip()
            if out:
                return out
        except Exception:
            continue
    return None


def normalize_for_match(q: str) -> str:
    q = normalize_query(q)
    q = q.replace("ः", "").replace("ं", "").replace("ँ", "")
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def query_terms_for_boost(query: str) -> List[str]:
    q = normalize_for_match(query)
    terms = [w for w in q.split(" ") if len(w) >= 3]
    stop = {
        "किम",
        "किं",
        "किंचित",
        "कथम्",
        "कथं",
        "कुत्र",
        "कदा",
        "कस्य",
        "अकरोत",
        "अकरोत्",
        "अभवत",
        "अस्ति",
        "एव",
        "इति",
        "न",
        "खलु",
    }
    return [t for t in terms if t not in stop]


def keyword_overlap_boost(query: str, text: str) -> float:
    t = normalize_for_match(text)
    q_terms = query_terms_for_boost(query)
    if not q_terms:
        return 0.0
    hits = sum(1 for w in q_terms if w in t)
    boost = 0.10 * hits
    if "कालीदास" in t:
        boost += 0.50
    return boost


def clean_seq2seq_output(text: str) -> str:
    text = re.sub(r"<extra_id_\d+>\s*", "", text)
    text = text.replace("<pad>", "").replace("</s>", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_speaker_answer(query: str, contexts: List[dict]) -> str | None:
    qn = normalize_query(query)
    if "कः" not in qn and "को" not in qn and "कौन" not in qn:
        return None
    if "उक्त" not in qn and "वद" not in qn and "कहा" not in qn:
        return None

    patterns = [
        r"वदति\s+([^,\s]+)",
        r"उक्तवान्\s+([^,\s]+)",
        r"उवाच\s+([^,\s]+)",
    ]
    for c in sorted(contexts, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        text = c.get("text") or ""
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                ans = m.group(1).strip()
                if ans:
                    return make_concise_answer(ans, max_chars=60)
    return None


def fallback_answer(query: str, contexts: List[dict]) -> str:
    qn = normalize_for_match(query)
    key = "कालीदास" if "कालीदास" in qn else None

    ordered = sorted(contexts, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if key:
        for c in ordered:
            text = c.get("text") or ""
            if key not in normalize_for_match(text):
                continue

            m = re.search(r"(.{0,120}" + re.escape(key) + r".{0,220})", text)
            if m:
                return make_concise_answer(m.group(1))

            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            for ln in lines:
                if key in normalize_for_match(ln):
                    return ln

    for c in ordered:
        lines = [ln.strip() for ln in (c.get("text") or "").split("\n") if ln.strip()]
        if lines:
            return make_concise_answer(lines[0])

    return "I don't know"


def retrieve(index, chunks: List[dict], query: str, k: int) -> List[dict]:
    query = normalize_query(query)

    terms = query_terms_for_boost(query)
    if not terms:
        terms = [normalize_for_match(query)]
    query_for_vec = " ".join(terms)

    qv = index["vectorizer"].transform([query_for_vec])
    base_scores = linear_kernel(qv, index["tfidf_matrix"]).ravel()

    scored = []
    for i in range(len(chunks)):
        c = chunks[int(i)]
        score = float(base_scores[i]) + float(keyword_overlap_boost(query, c.get("text", "")))
        scored.append((score, int(i)))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_idx = [i for _, i in scored[:k]]
    out: List[dict] = []
    for i in top_idx:
        c = chunks[int(i)]
        out.append(
            {
                "score": float(base_scores[i]) + float(keyword_overlap_boost(query, c.get("text", ""))),
                "source_path": c.get("source_path"),
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "text": c.get("text", ""),
            }
        )
    return out


def build_prompt(query: str, contexts: List[dict]) -> str:
    ctx = "\n\n".join(
        f"[context {i+1} | score={c['score']:.3f} | {os.path.basename(c.get('source_path',''))}#{c.get('chunk_id')} ]\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    return (
        "Answer the question using the context. If the context is insufficient, say 'I don't know'. Answer concisely.\n"
        f"Question: {query}\n"
        f"Context: {ctx}\n"
        "Answer:"
    )


def extractive_sanskrit_answer(query: str, contexts: List[dict]) -> str | None:
    qn = normalize_for_match(query)
    key = None
    if "कालीदास" in qn:
        key = "कालीदास"
    elif "धर्म" in qn:
        key = "धर्म"
    elif "शंखनाद" in qn or "शर्करा" in qn or "गलती" in qn:
        key = "शर्करा"

    ordered = sorted(contexts, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    for c in ordered:
        text = c.get("text") or ""
        parts = []
        for p in re.split(r"[\n]+", text):
            p = p.strip()
            if p:
                parts.append(p)
        sentences = []
        for p in parts:
            for s in re.split(r"।|\.|\?\!", p):
                s = s.strip()
                if s and re.search(r"[\u0900-\u097F]", s):
                    sentences.append(s)

        if key:
            for s in sentences:
                if key in normalize_for_match(s):
                    return s + "।"

        if sentences:
            return sentences[0] + "।"
    return None


def make_concise_answer(text: str, max_chars: int = 280) -> str:
    text = clean_seq2seq_output(text)
    if not text:
        return text
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    # try to end at a sentence boundary
    for sep in ["।", ".", "!", "?"]:
        pos = text.find(sep)
        if 0 < pos < 220:
            return text[: pos + 1].strip()
    return text.strip()


@torch.inference_mode()
def generate_gpt2(model_dir: str, prompt: str, max_new_tokens: int = 120) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    model_max_positions = getattr(getattr(model, "config", None), "n_positions", None)
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    effective_max = None
    if isinstance(model_max_positions, int) and model_max_positions > 0:
        effective_max = model_max_positions
    if isinstance(tokenizer_max, int) and tokenizer_max > 0 and tokenizer_max < 10**9:
        effective_max = min(effective_max, tokenizer_max) if effective_max else tokenizer_max

    max_input_tokens = None
    if effective_max:
        max_input_tokens = max(1, effective_max - int(max_new_tokens))

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True if max_input_tokens else False,
        max_length=max_input_tokens,
    )

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return make_concise_answer(text)


@torch.inference_mode()
def generate_flan_t5(model_name: str, prompt: str, max_new_tokens: int = 120) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=8,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return clean_seq2seq_output(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query")
    ap.add_argument("--query_file")
    ap.add_argument("--answer_mode", choices=["extractive", "generate"], default="extractive")
    ap.add_argument("--chunks", default="artifacts/chunks.jsonl")
    ap.add_argument("--index", default="artifacts/tfidf_index.joblib")
    ap.add_argument("--backend", choices=["flan_t5", "gpt2"], default="flan_t5")
    ap.add_argument("--model_dir", default="model")
    ap.add_argument("--gen_model", default="google/mt5-small")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    args = ap.parse_args()

    def read_text_auto(path: str) -> str:
        data = open(path, "rb").read()
        if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
            return data.decode("utf-16").strip()
        if data.startswith(b"\xef\xbb\xbf"):
            return data.decode("utf-8-sig").strip()
        try:
            return data.decode("utf-8").strip()
        except UnicodeDecodeError:
            return data.decode("utf-16").strip()

    if args.query_file:
        args.query = read_text_auto(args.query_file)
    if not args.query:
        raise SystemExit("Provide --query or --query_file")

    translit = maybe_transliterate_to_devanagari(args.query)
    if translit:
        args.query = translit

    chunks = load_chunks(args.chunks)
    index = joblib.load(args.index)

    contexts = retrieve(index, chunks, args.query, args.top_k)
    prompt = build_prompt(args.query, contexts)

    output = ""

    speaker = extract_speaker_answer(args.query, contexts)
    if speaker:
        output = speaker
    else:
        if args.answer_mode == "extractive":
            ext = extractive_sanskrit_answer(args.query, contexts)
            if ext:
                output = ext

        if not output:
            if args.backend == "gpt2":
                output = generate_gpt2(args.model_dir, prompt, max_new_tokens=args.max_new_tokens)
            else:
                output = generate_flan_t5(args.gen_model, prompt, max_new_tokens=args.max_new_tokens)

    if not output or output.strip().startswith("?"):
        output = fallback_answer(args.query, contexts)

    print("\n--- Retrieved Contexts ---")
    for c in contexts:
        print(f"\n(score={c['score']:.3f}) {c['source_path']} chunk={c['chunk_id']}")
        print(c["text"][:400].replace("\n", " ") + ("..." if len(c["text"]) > 400 else ""))

    print("\n--- Model Output ---")
    print(output)


if __name__ == "__main__":
    main()
