from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Union
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class QdrantService:
    # init / collection bootstrap
    def __init__(
        self,
        dimension: int,
        collection_name: str = "documents",
        recreate_collection: bool = False,
        distance: rest.Distance = rest.Distance.COSINE,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация QdrantService…")

        host = os.getenv("QDRANT_HOST", "qdrant")
        port = int(os.getenv("QDRANT_PORT", "6333"))

        self.client = QdrantClient(host=host, port=port, timeout=300.0)
        self.collection_name = collection_name
        self.dimension = dimension

        if recreate_collection:
            self.logger.info("⇢ пересоздаём коллекцию '%s'", collection_name)
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=dimension, distance=distance),
            )
        else:
            try:
                self.client.get_collection(collection_name)
                self.logger.info("Коллекция '%s' уже существует — пропускаем создание", collection_name)
            except Exception:
                self.logger.info("⇢ коллекция не найдена — создаём новую")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(size=dimension, distance=distance),
                )

        self.logger.info(
            "QdrantService готов: host=%s | port=%s | collection=%s | dim=%d",
            host,
            port,
            collection_name,
            dimension,
        )

    # existence checks
    def check_document_exists_by_original_id(self, original_news_id: str) -> bool:
        """
        True, если точка с payload.original_id уже в базе.
        """
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=rest.Filter(
                    must=[rest.FieldCondition(key="original_id", match=rest.MatchValue(value=original_news_id))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return bool(points)
        except Exception as e:
            self.logger.error("Ошибка проверки original_id=%s: %s", original_news_id, e)
            return False

    def check_document_exists(self, point_id: str) -> bool:
        """
        True, если в коллекции уже есть точка с заданным point_id.
        """
        try:
            res = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=False,
                with_payload=False,
            )
            return bool(res)
        except Exception as e:
            self.logger.error("Ошибка проверки point_id=%s: %s", point_id, e)
            return False

    # upload helpers
    def upload_document(
        self,
        embedding: Sequence[float],
        text: str,
        timestamp: int,
        extra_payload: Optional[Dict] = None,
        point_id: Optional[str] = None,
    ) -> str:
        """
        Добавляет один документ (чанк). Если point_id не задан — сгенерирует UUID.
        """
        pid = point_id or str(uuid4())

        payload: Dict = {"text": text, "timestamp": timestamp}
        if extra_payload:
            payload.update(extra_payload)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                rest.PointStruct(id=pid, vector=list(embedding), payload=payload),
            ],
        )
        self.logger.info("↑ добавлен point_id=%s | ts=%s", pid, timestamp)
        return pid

    def upload_documents_batch(
        self,
        embeddings: List[Sequence[float]],
        texts: List[str],
        timestamps: List[int],
        extra_payloads: Optional[List[Dict]] = None,
        point_ids: Optional[List[str]] = None,
        batch_size: int = 128,
    ) -> List[str]:
        """
        Пакетная загрузка (чтобы существенно ускорить заливку большого объёма).
        """
        n = len(embeddings)
        if not (n == len(texts) == len(timestamps)):
            raise ValueError("embeddings, texts и timestamps должны быть одинаковой длины")

        pids: List[str] = point_ids or [str(uuid4()) for _ in range(n)]

        points: List[rest.PointStruct] = []
        for i in range(n):
            pl = {"text": texts[i], "timestamp": timestamps[i]}
            if extra_payloads and i < len(extra_payloads):
                pl.update(extra_payloads[i])
            points.append(rest.PointStruct(id=pids[i], vector=list(embeddings[i]), payload=pl))

        for chunk_start in range(0, n, batch_size):
            chunk = points[chunk_start : chunk_start + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=chunk)
            self.logger.info("↑ batch %d-%d / %d загружен", chunk_start + 1, min(chunk_start + batch_size, n), n)

        return pids

    # search helpers
    def search_embeddings(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        params = dict(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        if score_threshold is not None:
            params["score_threshold"] = score_threshold
        return self.client.search(**params)

    def search_with_time_filter(
        self,
        query_embedding: Sequence[float],
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """
        Векторный поиск + фильтр по полю `timestamp`.
        Любая из границ может быть None.
        """
        conditions: List[rest.FieldCondition] = []

        if from_ts is not None:
            conditions.append(rest.FieldCondition(key="timestamp", range=rest.Range(gte=from_ts)))
        if to_ts is not None:
            conditions.append(rest.FieldCondition(key="timestamp", range=rest.Range(lte=to_ts)))

        params = dict(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        if conditions:
            params["query_filter"] = rest.Filter(must=conditions)  #  ← важно: **query_filter**, не filter
        if score_threshold is not None:
            params["score_threshold"] = score_threshold

        return self.client.search(**params)

    # misc helpers
    def get_collection_info(self) -> Optional[Dict]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": info.status,
                "points": info.points_count,
                "vectors": info.vectors_count,
                "indexed_vectors": info.indexed_vectors_count,
                "segments": info.segments_count,
            }
        except Exception as e:
            self.logger.error("Не удалось получить информацию о коллекции: %s", e)
            return None
