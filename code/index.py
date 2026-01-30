from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class IndexArtifacts:
    vectorizer: TfidfVectorizer
    tfidf_matrix: Any
    metas: List[dict]


def load_chunks(path: str) -> Tuple[List[str], List[dict]]:
    texts: List[str] = []
    metas: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append({k: obj[k] for k in obj.keys() if k != "text"})
    return texts, metas


def build_index(texts: List[str], metas: List[dict]) -> IndexArtifacts:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=False,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return IndexArtifacts(vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, metas=metas)


def save_index(art: IndexArtifacts, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(
        {
            "vectorizer": art.vectorizer,
            "tfidf_matrix": art.tfidf_matrix,
            "metas": art.metas,
        },
        out_path,
    )


def load_index(path: str):
    return joblib.load(path)


def search(index, query: str, k: int = 4):
    qv = index["vectorizer"].transform([query])
    scores = linear_kernel(qv, index["tfidf_matrix"]).ravel()
    top_idx = scores.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        results.append({"score": float(scores[i]), "meta": index["metas"][int(i)]})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="artifacts/chunks.jsonl")
    ap.add_argument("--out", default="artifacts/tfidf_index.joblib")
    args = ap.parse_args()

    texts, metas = load_chunks(args.chunks)
    art = build_index(texts, metas)
    save_index(art, args.out)
    print(f"Saved index to {args.out} (chunks={len(texts)})")


if __name__ == "__main__":
    main()
