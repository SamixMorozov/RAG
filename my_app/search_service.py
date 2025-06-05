import logging
from datetime import datetime
from typing import List, Dict, Optional

from .llm_client import LLMClient
from .qdrant_client import QdrantService

logger = logging.getLogger(__name__)


class SearchService:
    """Основной сервис RAG-поиска по базе новостей."""

    def __init__(self, embedding_model, qdrant_service: QdrantService):
        self.embedding_model = embedding_model
        self.qdrant_service = qdrant_service
        self.llm_client = LLMClient()
        self.top_k = 5

        logger.info("Инициализация SearchService завершена")
        logger.debug("Параметры: top_k=%s", self.top_k)

    def answer_question(self, question: str) -> str:
        """
        Полный цикл:
        1) анализ вопроса (LLM + модерация),
        2) поиск релевантных чанков,
        3) выбор документов,
        4) генерация ответа,
        5) валидация ответа.
        """

        logger.info("Начало обработки вопроса: '%s'", question)

        try:
            analysis = self.llm_client.analyze_question(question)
            if analysis.get("error"):
                # LLM вернул ошибку модерации
                return "Извините, я не могу ответить на ваш вопрос"

            rephrased_q = analysis.get("rephrased_question") or question
            logger.info(
                "Переформулированный вопрос: '%s'; даты: start=%s, end=%s",
                rephrased_q,
                analysis.get("start_date"),
                analysis.get("end_date"),
            )

            chunks = self._search_documents(rephrased_q, analysis)
            logger.info("Найдено чанков: %d", len(chunks))
            if not chunks:
                return "Извините, я не могу ответить на ваш вопрос"

            selected_doc_ids = self.llm_client.select_docs(rephrased_q, chunks)
            logger.info("Выбранные doc_id: %s", selected_doc_ids)
            if not selected_doc_ids:
                return "Извините, я не могу ответить на ваш вопрос"

            full_article_chunks = [
                c for c in chunks if c.get("doc_id") in selected_doc_ids
            ]
            if not full_article_chunks:
                return "Извините, я не могу ответить на ваш вопрос"

            full_article_chunks.sort(
                key=lambda c: (c.get("doc_id"), c.get("chunk_index", 0))
            )

            logger.info(
                "Передаём в LLM %d чанков (первый doc_id=%s, chunk=%s)",
                len(full_article_chunks),
                full_article_chunks[0].get("doc_id"),
                full_article_chunks[0].get("chunk_index"),
            )

            final_answer, is_valid = self.llm_client.generate_answer(
                question,
                full_article_chunks,
                temperature=0.3,
                max_tokens=500,
                validate_answer=True,
            )

            logger.info("Сгенерированный ответ: '%s' | валиден: %s", final_answer, is_valid)

            if not is_valid:
                return "Извините, я не могу ответить на ваш вопрос"

            return final_answer

        except Exception as exc:
            logger.error("Критическая ошибка обработки: %s", exc, exc_info=True)
            return "Извините, я не могу ответить на ваш вопрос"

    def _search_documents(self, question: str, analysis: Dict) -> List[Dict]:
        """Получаем top-k чанков из Qdrant (с учётом дат, если заданы)."""
        try:
            query_embedding = self.embedding_model.get_embeddings([question])[0].tolist()
            time_filter = self._create_time_filter(analysis)

            if time_filter:
                logger.info(
                    "Поиск с фильтром: %s — %s",
                    datetime.fromtimestamp(time_filter["from_ts"]).isoformat(),
                    datetime.fromtimestamp(time_filter["to_ts"]).isoformat(),
                )
                results = self.qdrant_service.search_with_time_filter(
                    query_embedding,
                    time_filter["from_ts"],
                    time_filter["to_ts"],
                    self.top_k,
                )
            else:
                logger.info("Поиск без временного фильтра")
                results = self.qdrant_service.search_embeddings(
                    query_embedding,
                    self.top_k,
                )

            logger.info("Найдено результатов в Qdrant: %d", len(results))

            return [self._parse_chunk(hit) for hit in results]

        except Exception as exc:
            logger.error("Ошибка поиска документов: %s", exc, exc_info=True)
            return []

    @staticmethod
    def _create_time_filter(analysis: Dict) -> Optional[Dict]:
        """Формируем диапазон timestamp (UTC, сек) из start_date / end_date."""
        try:
            from_ts = to_ts = None

            if analysis.get("start_date"):
                start_dt = datetime.fromisoformat(
                    analysis["start_date"].replace("Z", "+00:00")
                )
                from_ts = int(start_dt.timestamp())

            if analysis.get("end_date"):
                end_dt = datetime.fromisoformat(
                    analysis["end_date"].replace("Z", "+00:00")
                ).replace(hour=23, minute=59, second=59)
                to_ts = int(end_dt.timestamp())

            # если указана только одна дата — ищем в пределах суток
            if from_ts and not to_ts:
                to_ts = (
                    datetime.fromtimestamp(from_ts)
                    .replace(hour=23, minute=59, second=59)
                    .timestamp()
                )
            elif to_ts and not from_ts:
                from_ts = (
                    datetime.fromtimestamp(to_ts)
                    .replace(hour=0, minute=0, second=0)
                    .timestamp()
                )

            if from_ts is None or to_ts is None:
                return None

            return {"from_ts": int(from_ts), "to_ts": int(to_ts)}

        except Exception as exc:
            logger.error("Ошибка time-filter: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _parse_chunk(point) -> Dict:
        """Приводим объект Qdrant Point → dict для LLM."""
        try:
            payload = point.payload or {}
            return {
                "doc_id": payload.get("doc_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "score": point.score,
                "title": payload.get("title", ""),
                "text": payload.get("text", ""),
                "url": payload.get("url", ""),
                "timestamp": payload.get("timestamp", 0),
            }
        except Exception as exc:
            logger.error("Ошибка парсинга чанка: %s", exc, exc_info=True)
            return {}
