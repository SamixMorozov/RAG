import logging
import re
from typing import List
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class TextSplitter:
    def __init__(
        self,
        small_chunk_size: int = 50,
        window_size_in_small_chunks: int = 16,
        min_remaining_chunks: int = 4,
        llm_client: LLMClient = None
    ):
        self.small_chunk_size = small_chunk_size
        self.window_size_in_small_chunks = window_size_in_small_chunks
        self.min_remaining_chunks = min_remaining_chunks
        self.llm_client = llm_client or LLMClient()

    @staticmethod
    def naive_tokenize(text: str) -> List[str]:
        return text.split()

    def small_chunk_splitter(self, text: str) -> List[str]:
        tokens = self.naive_tokenize(text)
        chunks = []
        start = 0
        chunk_index = 1

        while start < len(tokens):
            end = start + self.small_chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            chunk_labeled = f"<|start_chunk_{chunk_index}|>\n{chunk_text}\n<|end_chunk_{chunk_index}|>"
            chunks.append(chunk_labeled)
            start = end
            chunk_index += 1

        return chunks

    def ask_for_splits(self, chunks_window: List[str]) -> List[int]:
        window_text = "\n".join(chunks_window)
        prompt = f"""
Вы — помощник, специализирующийся на разбиении текста на тематически связанные секции.
Ниже представлен текст, уже разделённый на «мелкие чанки», каждый помечен тегами <|start_chunk_X|> и <|end_chunk_X|>, где X — номер чанка.

Ваша задача — определить, после каких чанков нужно сделать разрыв, чтобы сгруппировать соседние чанки похожих тем вместе.
Ответьте в формате:
split_after: X, Y, Z
где X, Y, Z — номера чанков, после которых нужно сделать разрыв (в порядке возрастания).
Не добавляйте лишних комментариев или текста, только эту строку.

Вот окно с чанками:

{window_text}
"""
        response = self.llm_client.call_ollama(prompt)
        logger.debug("Ответ LLM на split_after:\n%s", response)

        match = re.search(r"split_after:\s*(.*)", response)
        if not match:
            logger.info("Не найдено разбиений, LLM вернула пустой ответ.")
            return []

        split_str = match.group(1)
        split_indexes = []
        for s in split_str.split(","):
            s = s.strip()
            if s.isdigit():
                split_indexes.append(int(s))
            else:
                logger.warning("Не удалось распознать индекс чанка: '%s'", s)

        return split_indexes

    @staticmethod
    def remove_chunk_tags(chunk_text: str) -> str:
        cleaned = re.sub(r"<\|start_chunk_\d+\|>", "", chunk_text)
        cleaned = re.sub(r"<\|end_chunk_\d+\|>", "", cleaned)
        return cleaned.strip()

    def semantic_chunk_text(self, text: str) -> List[str]:
        logger.info("Запуск семантического чанкинга. Длина текста: %d символов", len(text))
        small_chunks = self.small_chunk_splitter(text)
        total_small_chunks = len(small_chunks)

        if total_small_chunks == 0:
            logger.info("Текст пустой после разбиения на мелкие чанки.")
            return []

        final_chunks = []
        current_index = 0

        while current_index < total_small_chunks:
            end_index = min(current_index + self.window_size_in_small_chunks, total_small_chunks)
            window_chunks = small_chunks[current_index:end_index]
            splits_local = self.ask_for_splits(window_chunks)

            if not splits_local:
                splits_local = [len(window_chunks)]

            splits_local = sorted(set(splits_local))
            start_chunk = 0
            for split_after in splits_local:
                if split_after > len(window_chunks):
                    logger.warning(
                        "Модель вернула split_after=%d, превышающий количество чанков в окне=%d",
                        split_after, len(window_chunks)
                    )
                    split_after = len(window_chunks)

                combined_text = [
                    self.remove_chunk_tags(window_chunks[i]).strip()
                    for i in range(start_chunk, split_after)
                ]
                final_chunks.append("\n".join(combined_text))
                start_chunk = split_after

            if start_chunk < len(window_chunks):
                leftover_texts = [
                    self.remove_chunk_tags(window_chunks[i])
                    for i in range(start_chunk, len(window_chunks))
                ]
                final_chunks.append("\n".join(leftover_texts))

            current_index = end_index
            remaining = total_small_chunks - current_index
            if 0 < remaining < self.min_remaining_chunks:
                leftover_texts = [
                    self.remove_chunk_tags(small_chunks[i])
                    for i in range(current_index, total_small_chunks)
                ]
                final_chunks.append("\n".join(leftover_texts))
                break

        logger.info("Семантический чанкинг завершен. Всего финальных чанков: %d", len(final_chunks))
        return final_chunks

    def split_text_with_context(self, text: str, chunk_size=None, max_chunk_size=None, overlap=None) -> List[str]:
        original_small_chunk_size = self.small_chunk_size
        original_window_size = self.window_size_in_small_chunks
        
        try:
            if chunk_size:
                self.small_chunk_size = max(10, chunk_size // 10)
                logger.info(f"Установлен размер малого чанка: {self.small_chunk_size} токенов")
                
            if max_chunk_size:
                self.window_size_in_small_chunks = max(4, max_chunk_size // self.small_chunk_size)
                logger.info(f"Установлен размер окна: {self.window_size_in_small_chunks} малых чанков")
            
            return self.semantic_chunk_text(text)
        finally:
            self.small_chunk_size = original_small_chunk_size
            self.window_size_in_small_chunks = original_window_size
