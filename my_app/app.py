from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from filelock import FileLock, Timeout
from gradio import mount_gradio_app

from my_app.embeddings import EmbeddingModel
from my_app.file_processor import FileProcessor
from my_app.qdrant_client import QdrantService
from my_app.data_ingestion import DataIngestion
from my_app.search_service import SearchService
from my_app.gradio_interface import create_gradio_interface

# ──────────────────────────  базовое логирование  ────────────────────────── #
logging.basicConfig(
    format="%(asctime)s  %(levelname)s - %(name)s: %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
logger = logging.getLogger(__name__)

# ─────────────────────────────  глобальные объекты  ──────────────────────── #
embedding_model = EmbeddingModel()
embedding_model.ensure_initialized()

file_processor = FileProcessor()
qdrant_service = QdrantService(
    dimension=embedding_model.get_dimension(),
    collection_name="documents",
    recreate_collection=False,
)

NEWS_JSON_PATH = Path(os.getenv("NEWS_JSON_PATH", "/app/news_db/news_data.json"))
FLAG_PATH = Path(os.getenv("FLAG_PATH", "/app/.ingested_ok"))
LOCK = FileLock("/app/.ingestion.lock", timeout=600)          # 10 мин

# ────────────────────────────────  FastAPI  ──────────────────────────────── #
app = FastAPI(title="RAG-demo")

async def _run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))

def _ingest_once() -> bool:
    """Блокирующая функция: безопасная однократная загрузка новостей."""
    if FLAG_PATH.exists():
        logger.info("Флаг .ingested_ok найден — пропускаем загрузку")
        return True
    if not NEWS_JSON_PATH.exists():
        logger.warning("Файл с новостями не найден — считаем, что нечего загружать")
        FLAG_PATH.touch()
        return True

    try:
        with LOCK:  # filelock гарантирует, что только один процесс войдёт сюда
            if FLAG_PATH.exists():
                return True

            ingestion = DataIngestion(embedding_model, file_processor, qdrant_service)
            logger.info("=== Ингест новостей → Qdrant ===")
            if ingestion.process_json_file(str(NEWS_JSON_PATH)):
                FLAG_PATH.touch()
                logger.info("✅ Ингест завершён")
                return True
            logger.error("❌ Ингест завершился ошибкой")
            return False
    except Timeout:
        logger.info("Другой процесс выполняет загрузку — ждём завершения…")
        LOCK.acquire()          # дождёмся освобождения
        LOCK.release()
        return FLAG_PATH.exists()

@app.on_event("startup")
async def startup():
    ok = await _run_blocking(_ingest_once)
    if not ok:
        logger.error("Данные не загружены — приложение стартует, но поиску нечего искать")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Завершение работы приложения")

@app.get("/")
async def root():
    return {"status": "ok", "gradio": "/gradio", "data_loaded": FLAG_PATH.exists()}

@app.get("/health")
async def health():
    info = qdrant_service.get_collection_info()
    return {
        "status": "healthy" if info else "degraded",
        "points": info.get("points_count") if info else 0,
        "data_ingested": FLAG_PATH.exists(),
        "device": embedding_model.get_device(),
    }

# ──────────────────────────────  Gradio UI  ──────────────────────────────── #
search_service = SearchService(embedding_model, qdrant_service)
app = mount_gradio_app(app, create_gradio_interface(search_service), path="/gradio")

# ──────────────────────────────  Uvicorn run  ────────────────────────────── #
if __name__ == "__main__":
    uvicorn.run(
        "my_app.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 58084)),
        log_level="info",
        timeout_keep_alive=1200,
        reload=False,
        workers=1,         # важно: один воркер, чтобы не плодить процессы
    )
