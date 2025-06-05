from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Union

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Singleton-обёртка над SentenceTransformer."""

    _instance: "EmbeddingModel | None" = None
    _lock = threading.Lock()

    #  Singleton                                                               #
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:          # double-checked-locking
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    #  Private helpers                                                         #
    def _initialize(self) -> None:
        if self._initialized:                      # уже инициализирована
            return

        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация модели эмбеддингов…")

        model_name = os.getenv("EMBEDDING_MODEL_NAME", "ai-forever/ru-en-RoSBERTa")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        start = time.time()
        self.embedding_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.logger.info(
            f"Модель «{model_name}» загружена за {time.time()-start:.2f} с, "
            f"устройство: {device}, dim={self.dimension}"
        )
        self._initialized = True

    #  Public API                                                              #
    def ensure_initialized(self) -> None:
        """Гарантирует, что модель готова (удобно звать из FastAPI startup)."""
        self._initialize()

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        *,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        """Кодирование строки/списка строк в тензор эмбеддингов."""
        self._initialize()

        if isinstance(texts, str):
            texts = [texts]

        cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not cleaned:
            self.logger.warning("Переданы пустые тексты; возвращаю []")
            return []

        return self.embedding_model.encode(
            cleaned,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(cleaned) > 100,
        )

    def get_single_embedding(self, text: str, *, normalize: bool = True):
        self._initialize()
        if not text.strip():
            self.logger.warning("Пустая строка → zero-vector")
            return torch.zeros(self.dimension)
        return self.get_embeddings([text], normalize=normalize)[0]

    def get_dimension(self) -> int:
        self._initialize()
        return self.dimension

    def get_device(self) -> str:                   # используется в health-чеке
        self._initialize()
        return str(self.embedding_model.device)
