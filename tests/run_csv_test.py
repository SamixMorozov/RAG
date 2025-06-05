import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client.http import models as rest

from my_app.embeddings import EmbeddingModel
from my_app.qdrant_client import QdrantService
from my_app.llm_client import LLMClient
from my_app.search_service import SearchService
from my_app.file_processor import FileProcessor
from my_app.data_ingestion import DataIngestion

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_PATH = "tests/input_test_data.csv"
TEXT_COLUMN = "A"
QUESTION_COLUMN = "B"
PERFECT_ANSWER_COLUMN = "C"
ANSWER_COLUMN = "D"
TOP_CHUNKS_COLUMN = "E"
SELECTED_CHUNK_COLUMN = "F"

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")


def main():
    logger.info("Запуск тестирования CSV")

    embedding_model = EmbeddingModel()
    qdrant = QdrantService(embedding_model.dimension)
    file_processor = FileProcessor()
    data_ingestion = DataIngestion(embedding_model, file_processor, qdrant)
    llm_client = LLMClient()
    search_service = SearchService(embedding_model, qdrant)

    try:
        if qdrant.client.collection_exists(qdrant.collection_name):
            qdrant.client.delete_collection(qdrant.collection_name)
            logger.info(f"Коллекция {qdrant.collection_name} удалена")
        qdrant.client.create_collection(
            collection_name=qdrant.collection_name,
            vectors_config=rest.VectorParams(
                size=embedding_model.dimension,
                distance=rest.Distance.COSINE
            )
        )
        logger.info("Коллекция успешно пересоздана")
    except Exception as e:
        logger.error(f"Ошибка при очистке коллекции: {e}")

    df = pd.read_csv(CSV_PATH, encoding="cp1251")

    total_rows = len(df)
    logger.info(f"Всего строк в CSV файле: {total_rows}")

    df = df.dropna(subset=[QUESTION_COLUMN])
    filtered_rows = total_rows - len(df)
    logger.info(f"Удалено {filtered_rows} строк с пустыми вопросами")
    logger.info(f"Осталось для обработки: {len(df)} вопросов")

    for col in [PERFECT_ANSWER_COLUMN, ANSWER_COLUMN, TOP_CHUNKS_COLUMN, SELECTED_CHUNK_COLUMN]:
        if col not in df.columns:
            df[col] = ""

    unique_texts = df[TEXT_COLUMN].dropna().drop_duplicates().reset_index(drop=True)
    logger.info(f"В наборе данных {len(unique_texts)} уникальных текстов для загрузки")

    logger.info("Загрузка уникальных документов в Qdrant")
    for idx, text in unique_texts.items():
        text_str = str(text).strip()
        if text_str and text_str.lower() != "nan":
            doc_id = f"test-doc-{idx}"
            item = {"id": doc_id, "title": "", "text": text_str, "timestamp": ""}
            try:
                data_ingestion.process_json_data([item])
            except Exception as e:
                logger.error(f"Ошибка при загрузке документа {doc_id}: {e}")

    logger.info("Обработка вопросов")
    for idx, row in df.iterrows():
        question = str(row.get(QUESTION_COLUMN, "")).strip()
        if not question or question.lower() == "nan":
            continue
        logger.info(f"\nВопрос {idx}: {question}")
        try:
            if llm_client.check_obscene(question):
                df.at[idx, ANSWER_COLUMN] = "Извините, я не могу ответить на ваш вопрос"
                df.at[idx, TOP_CHUNKS_COLUMN] = "Вопрос отклонен"
                df.at[idx, SELECTED_CHUNK_COLUMN] = "Вопрос отклонен"
                continue
            analysis = llm_client.analyze_question(question)
            rephrased_question = analysis.get("rephrased_question") or question
            query_embedding = embedding_model.get_embeddings([rephrased_question])[0].tolist()
            time_filter = None
            if analysis.get("start_date") or analysis.get("end_date"):
                time_filter = search_service._create_time_filter(analysis)
            if time_filter:
                try:
                    search_results = qdrant.search_with_time_filter(
                        query_embedding,
                        time_filter.get("from_ts"),
                        time_filter.get("to_ts"),
                        search_service.top_k
                    )
                except Exception:
                    search_results = qdrant.search_embeddings(query_embedding, search_service.top_k)
            else:
                search_results = qdrant.search_embeddings(query_embedding, search_service.top_k)
            chunks = [search_service._parse_chunk(hit) for hit in search_results]
            if chunks:
                top_chunks_details = [
                    f"{c.get('doc_id','N/A')} ({c.get('score',0):.2f}): {c.get('text','').strip()}"
                    for c in chunks
                ]
                top_chunks_str = "\n---\n".join(top_chunks_details)
            else:
                top_chunks_str = "Не найдено"
            df.at[idx, TOP_CHUNKS_COLUMN] = top_chunks_str
            answer = search_service.answer_question(question)
            df.at[idx, ANSWER_COLUMN] = answer
            selected_ids = llm_client.select_docs(rephrased_question, chunks) if chunks else []
            if selected_ids:
                texts = []
                for sid in selected_ids:
                    sel_chunks = [c for c in chunks if c.get("doc_id") == sid]
                    if sel_chunks:
                        texts.append(f"{sid}: {sel_chunks[0].get('text','')}")
                df.at[idx, SELECTED_CHUNK_COLUMN] = " | ".join(texts)
            else:
                df.at[idx, SELECTED_CHUNK_COLUMN] = "Не выбраны"
            logger.info(f"Топ чанки: {top_chunks_str[:200]}...")
            logger.info(f"Выбранные документы: {', '.join(selected_ids) if selected_ids else 'нет'}")
            logger.info(f"Ответ: {answer}\n")
        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса {idx}: {e}")
            df.at[idx, ANSWER_COLUMN] = f"Ошибка: {str(e)}"
            df.at[idx, TOP_CHUNKS_COLUMN] = "Ошибка"
            df.at[idx, SELECTED_CHUNK_COLUMN] = "Ошибка"

    result_df = df[
        [
            TEXT_COLUMN,
            QUESTION_COLUMN,
            PERFECT_ANSWER_COLUMN,
            ANSWER_COLUMN,
            TOP_CHUNKS_COLUMN,
            SELECTED_CHUNK_COLUMN
        ]
    ]
    result_df.columns = [
        "Text",
        "Question",
        "Perfect_Answer",
        "Answer",
        "Top_Chunks",
        "Selected_Chunk"
    ]

    llm_name = (LLM_MODEL_NAME or "llm").replace("/", "_").replace(":", "_")
    emb_name = (EMBEDDING_MODEL_NAME or "embedding").replace("/", "_").replace(":", "_")

    result_dir = Path("tests/results")
    result_dir.mkdir(exist_ok=True, parents=True)

    results_path = result_dir / f"{llm_name}_{emb_name}_results.csv"
    result_df.to_csv(results_path, index=False, encoding="cp1251")
    logger.info(f"Результаты сохранены в {results_path}")

    try:
        if qdrant.client.collection_exists(qdrant.collection_name):
            qdrant.client.delete_collection(qdrant.collection_name)
            logger.info(f"Коллекция {qdrant.collection_name} успешно очищена после тестирования")
    except Exception as e:
        logger.error(f"Ошибка при очистке коллекции: {e}")


if __name__ == "__main__":
    main()
