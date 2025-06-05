import os
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd

from my_app.file_processor import FileProcessor
from my_app.text_splitter import TextSplitter
from my_app.search_service import SearchService


class DataIngestion:
    def __init__(self, embedding_model, file_processor: FileProcessor, qdrant_service):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        self.file_processor = file_processor
        self.qdrant_service = qdrant_service
        self.splitter = TextSplitter()

    def process_json_file(self, file_path: str) -> bool:
        try:
            self.logger.info(f"Чтение JSON-файла: {file_path}")
            if not os.path.exists(file_path):
                self.logger.error(f"Файл {file_path} не найден.")
                return False

            news_data = self.file_processor.read_json_file_sync(file_path)
            return self.process_json_data(news_data)

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных из {file_path}: {e}", exc_info=True)
            return False

    def process_json_data(self, news_data: List[Dict]) -> bool:
        try:
            self.logger.info(f"Обработка данных напрямую (кол-во записей: {len(news_data)})")
            if not news_data:
                self.logger.error("Список данных пуст.")
                return False

            count_new = 0

            for idx, item in enumerate(news_data):
                news_id = item.get("id", "")
                if not news_id:
                    self.logger.warning(f"Запись #{idx} не имеет поля 'id', пропускаем.")
                    continue

                if self.qdrant_service.check_document_exists_by_original_id(news_id):
                    self.logger.info(f"Новость id={news_id} уже обработана, пропускаем.")
                    continue

                title = item.get("title", "")
                text = item.get("text", "")
                full_text = f"{title}\n\n{text}".strip()

                chunks = self.splitter.split_text_with_context(full_text, max_chunk_size=500, overlap=100)
                self.logger.info(f"Новость '{news_id}' разбита на {len(chunks)} чанков.")

                for chunk_index, chunk in enumerate(chunks):
                    embedding_tensor = self.embedding_model.get_embeddings([chunk])[0]
                    embedding = embedding_tensor.tolist() if hasattr(embedding_tensor, 'tolist') else embedding_tensor
                    unix_ts = self._convert_iso_to_unix(item.get("timestamp", ""))

                    payload = {
                        "doc_id": news_id,
                        "original_id": news_id,  # Важно! Сохраняем оригинальный ID
                        "chunk_index": chunk_index,
                        "title": title,
                        "author": item.get("author", ""),
                        "url": item.get("url", ""),
                        "text": chunk,
                        "timestamp": unix_ts
                    }

                    chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{news_id}_{chunk_index}"))

                    self.qdrant_service.upload_document(
                        embedding=embedding,
                        text=chunk,
                        timestamp=unix_ts,
                        extra_payload=payload,
                        point_id=chunk_uuid
                    )

                    count_new += 1
                    self.logger.info(
                        f"({idx+1}/{len(news_data)}) Чанк {chunk_index} новости '{news_id}' загружен в Qdrant."
                    )

            self.logger.info(f"Всего новых чанков загружено: {count_new}.")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            return False

    def _convert_iso_to_unix(self, value) -> int:
        if value is None:
            return 0
        try:
            import math
            if isinstance(value, float) and math.isnan(value):
                return 0
        except Exception:
            pass
        if isinstance(value, (int, float)):
            if value > 1e12:
                value /= 1000.0
            return int(value)
        if isinstance(value, datetime):
            return int(value.timestamp())
        iso_str = str(value).strip()
        if not iso_str:
            return 0
        try:
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp())
        except ValueError:
            self.logger.warning(f"Невалидный ISO-формат: {iso_str}, ставим 0.")
            return 0

    def process_csv_for_test(self, csv_path: str, output_path: str, search_service: SearchService) -> bool:
        try:
            self.logger.info(f"Чтение тестового CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            if 'A' not in df.columns or 'B' not in df.columns:
                self.logger.error("Ожидаются колонки 'A' (новость) и 'B' (вопрос)")
                return False

            answers = []
            for idx, row in df.iterrows():
                question = str(row['B'])
                self.logger.info(f"Обработка вопроса {idx+1}/{len(df)}: {question}")
                try:
                    answer = search_service.answer_question(question)
                except Exception as e:
                    self.logger.warning(f"Ошибка при обработке вопроса '{question}': {e}")
                    answer = "Ошибка"

                answers.append(answer)

            df['D'] = answers
            df.to_csv(output_path, index=False)
            self.logger.info(f"Тестовый CSV сохранён: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при обработке CSV-файла: {e}", exc_info=True)
            return False
