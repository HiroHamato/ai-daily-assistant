from __future__ import annotations

import asyncio
import json
import os
import logging
from typing import List, Tuple

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging


logger = logging.getLogger("vector_store.faiss")


class FaissStore:
    def __init__(self, index_path: str, meta_path: str, model_name: str) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_name = model_name

        self.model: SentenceTransformer | None = None
        self.dimension: int | None = None
        self.index: faiss.Index | None = None
        self.metadata: List[str] = []

        self._lock = asyncio.Lock()

    async def init(self) -> None:
        logger.info("Initializing FAISS store: model=%s", self.model_name)
        # Enable HF tqdm progress and verbose logs
        os.environ.setdefault("HF_HUB_ENABLE_TQDM", "1")
        hf_logging.set_verbosity_info()

        def _load() -> tuple[SentenceTransformer, int, faiss.Index, List[str]]:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            model = SentenceTransformer(self.model_name)
            dimension = model.get_sentence_embedding_dimension()
            logger.info("Model loaded: dim=%d", dimension)

            index = faiss.IndexFlatIP(dimension)
            metadata: List[str] = []
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                logger.info("Reading existing FAISS index from %s and metadata from %s", self.index_path, self.meta_path)
                try:
                    index = faiss.read_index(self.index_path)
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    logger.info("Index loaded: ntotal=%d, meta=%d", index.ntotal, len(metadata))
                except Exception as e:
                    logger.warning("Failed to read existing index/meta: %s. Starting fresh.", e)
                    metadata = []
            else:
                logger.info("No existing index found. Starting with a fresh index.")
            return model, dimension, index, metadata

        self.model, self.dimension, self.index, self.metadata = await asyncio.to_thread(_load)
        logger.info("FAISS store initialized.")

    async def _persist(self) -> None:
        def _write(index: faiss.Index, metadata: List[str]) -> None:
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            faiss.write_index(index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        assert self.index is not None
        await asyncio.to_thread(_write, self.index, self.metadata)
        logger.info("Persisted FAISS index and metadata: ntotal=%d, meta=%d", self.index.ntotal, len(self.metadata))

    async def add_texts(self, texts: List[str]) -> None:
        if not texts:
            return
        assert self.model is not None
        assert self.index is not None

        logger.info("Encoding %d texts for KB add...", len(texts))
        embeddings = await asyncio.to_thread(self.model.encode, texts, normalize_embeddings=True)
        async with self._lock:
            self.index.add(embeddings)
            self.metadata.extend(texts)
            await self._persist()
        logger.info("Added %d texts to KB. New ntotal=%d", len(texts), self.index.ntotal)

    async def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        assert self.model is not None
        assert self.index is not None
        if self.index.ntotal == 0:
            logger.info("Search skipped: empty index")
            return []

        logger.info("Encoding query for KB search: '%s'", query)
        q_emb = await asyncio.to_thread(self.model.encode, [query], normalize_embeddings=True)

        def _search(index: faiss.Index, q, topk: int):
            scores, idxs = index.search(q, topk)
            return scores, idxs

        scores, idxs = await asyncio.to_thread(_search, self.index, q_emb, k)
        results: List[Tuple[str, float]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append((self.metadata[i], float(s)))
        logger.info("KB search returned %d results", len(results))
        return results
