from __future__ import annotations

import json
import os
from typing import List, Tuple

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer


class FaissStore:
    def __init__(self, index_path: str, meta_path: str, model_name: str) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[str] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_texts(self, texts: List[str]) -> None:
        if not texts:
            return
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.index.add(embeddings)
        self.metadata.extend(texts)
        self._persist()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb, k)
        results: List[Tuple[str, float]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append((self.metadata[i], float(s)))
        return results
