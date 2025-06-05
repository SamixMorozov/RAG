import logging
import json
import os
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("LLM_MODEL_NAME")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"


class LLMClient:
    def __init__(self):
        self.logger = logger
        self.api_url = OLLAMA_URL
        self.model = OLLAMA_MODEL
        self.logger.info(f"Использование Ollama API по адресу {self.api_url}, модель={self.model}")

    def call_ollama(self, prompt: str, temperature: float = 0.4, max_tokens: int = 200) -> Optional[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature, "max_tokens": max_tokens},
            "stream": False,
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=60, stream=False)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()
            self.logger.debug("Ответ модели: %s", raw)
            return raw
        except Exception as e:
            self.logger.error(f"Ошибка при вызове Ollама: {e}", exc_info=True)
            return None

    def _parse_json_from_llm_response(self, raw_answer: str) -> Dict[str, Any]:
        if not raw_answer:
            self.logger.warning("Получен пустой ответ от LLM")
            return {}
        cleaned = raw_answer.strip()
        if "```" in cleaned:
            try:
                start_idx = cleaned.find("```") + 3
                if cleaned[start_idx:].startswith("json"):
                    start_idx += 4
                end_idx = cleaned.rfind("```")
                if end_idx > start_idx:
                    cleaned = cleaned[start_idx:end_idx].strip()
            except Exception as e:
                self.logger.warning(f"Ошибка при удалении markdown-разметки: {e}")
        try:
            start_brace = cleaned.find("{")
            end_brace = cleaned.rfind("}")
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                cleaned = cleaned[start_brace : end_brace + 1]
        except Exception as e:
            self.logger.warning(f"Ошибка при извлечении фигурных скобок: {e}")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка при разборе JSON: {e}, ответ: {raw_answer!r}")
            return {}

    def check_obscene(self, question: str) -> bool:
        """
        Проверяет вопрос на нецензурную лексику и неприемлемый контент.
        Возвращает True, если контент неприемлем.
        """
        self.logger.info("Проверка вопроса на неприемлемый контент")
        prompt = f"""
Определи, содержит ли вопрос пользователя нецензурные, оскорбительные или неприемлемые выражения:
Вопрос: "{question}"
Ответь одним словом: "Нецензурно" или "Ок"
        """
        result = self.call_ollama(prompt, temperature=0.2, max_tokens=10)
        is_obscene = result and "Нецензурно" in result

        if is_obscene:
            self.logger.warning(f"Обнаружен неприемлемый контент в вопросе: {question}")
        else:
            self.logger.info("Вопрос прошел проверку на неприемлемый контент")

        return is_obscene

    def validate_final_answer(self, question: str, answer: str) -> bool:
        """
        Проверяет соответствие ответа вопросу.
        Возвращает True, если ответ приемлем.
        """
        self.logger.info("Валидация финального ответа")
        prompt = f"""
Проанализируй, насколько корректно ответ соответствует вопросу:
Вопрос: "{question}"
Ответ: "{answer}"

Учти, что ответ "В документе нет информации для ответа на этот вопрос" считается приемлемым.
Также учти, что ответ может быть неполным, но всё же подходящим.

Оцени, насколько ответ релевантен вопросу по шкале от 1 до 5, где:
1 - Абсолютно не соответствует
5 - Полностью соответствует

Ответь ТОЛЬКО числом от 1 до 5.
        """
        result = self.call_ollama(prompt, temperature=0.2, max_tokens=10)
        if not result:
            self.logger.warning("Не удалось получить оценку валидации")
            return False
        try:
            score = int(result.strip())
            is_valid = score >= 3
            self.logger.info(f"Оценка валидации ответа: {score}/5, валиден: {is_valid}")
            return is_valid
        except ValueError:
            self.logger.warning(f"Не удалось распарсить оценку валидации: {result}")
            return True

    def analyze_question(self, question: str) -> Dict[str, Optional[str]]:
        """
        Анализирует и переформулирует вопрос с извлечением дат.
        Включает проверку на неприемлемый контент.
        """
        self.logger.info("Анализ вопроса: переформулировка + временные рамки")

        # Проверяем вопрос на неприемлемый контент
        if self.check_obscene(question):
            self.logger.warning("Вопрос не прошел проверку на неприемлемый контент")
            return {
                "rephrased_question": None,
                "start_date": None,
                "end_date": None,
                "error": "Вопрос содержит неприемлемый контент"
            }

        current_time_iso = datetime.utcnow().isoformat()
        prompt = f"""
Сейчас {current_time_iso}.
Пользователь задал вопрос: "{question}"

Твоя задача строго следующая:
1) Переформулируй вопрос, сохранив его смысл и сделав формулировку максимально понятной и однозначной.
2) Извлеки дату или диапазон дат ТОЛЬКО если пользователь ЯВНО указал их в вопросе
   (например: "с 1 по 5 июня 2024 года", "вчера", "2023-03-15").
   • Если дата явно не указана, НЕ ПРИДУМЫВАЙ её.
   • В этом случае поля start_date и end_date ДОЛЖНЫ быть пустыми строками.

Верни ответ СТРОГО в формате JSON без пояснений:
{{
  "rephrased_question": "...",
  "start_date": "...",
  "end_date": "..."
}}

Если даты нет, верни:
{{
  "rephrased_question": "...",
  "start_date": "",
  "end_date": ""
}}
"""
        raw_answer = self.call_ollama(prompt, max_tokens=200)
        if not raw_answer:
            return {"rephrased_question": None, "start_date": None, "end_date": None}
        parsed = self._parse_json_from_llm_response(raw_answer)
        return {
            "rephrased_question": parsed.get("rephrased_question", "").strip() or None,
            "start_date": parsed.get("start_date", "").strip() or None,
            "end_date": parsed.get("end_date", "").strip() or None,
        }

    def select_docs(self, question: str, docs: List[Dict]) -> List[str]:
        chunks_info = "\n".join(
            [
                f"({d.get('doc_id','N/A')}, {d.get('chunk_index','N/A')}, {d.get('text','')})"
                for d in docs
            ]
        )
        prompt = f"""
Ты — система, отвечающая за отбор **только самых релевантных** документов для ответа на вопрос пользователя.

Вопрос пользователя:
{question}

Ниже приведены фрагменты документов. Каждый фрагмент имеет формат:
(doc_id, chunk_index, text)

{chunks_info}

Инструкция:
1. Внимательно проанализируй все фрагменты и определи, какая информация **необходима и достаточна** для полного и точного ответа на вопрос.
2. Выбери **только те** документы (doc_id), которые содержат **ключевую информацию**, напрямую отвечающую на вопрос.
   • Если один документ полностью покрывает вопрос — выбери **только его**.
   • Если для ответа требуется информация из нескольких документов — выбери **только их**.
   • Если **ни один** документ не содержит достаточно информации для ответа — верни **пустой список**.

Верни результат в виде **строгого JSON-объекта** без пояснений, строго по формату:
{{ "selected_doc_ids": ["doc_1", "doc_2"] }}
        """
        raw_answer = self.call_ollama(prompt, max_tokens=200)
        if not raw_answer:
            return []
        parsed = self._parse_json_from_llm_response(raw_answer)
        ids = parsed.get("selected_doc_ids", [])
        return [str(i).strip() for i in ids if str(i).strip()]

    def generate_answer(
        self,
        original_question: str,
        context_docs: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        validate_answer: bool = True,
        max_retries: int = 2
    ) -> Tuple[str, bool]:
        """
        Генерирует ответ с возможностью валидации.
        Возвращает кортеж (ответ, валиден_ли_ответ).
        """
        self.logger.debug(f"Генерируем ответ на основе {len(context_docs)} чанков документа")
        if context_docs:
            self.logger.debug(
                f"Первый чанк: doc_id={context_docs[0].get('doc_id')}, chunk_index={context_docs[0].get('chunk_index')}"
            )
            if len(context_docs) > 1:
                self.logger.debug(
                    f"Последний чанк: doc_id={context_docs[-1].get('doc_id')}, chunk_index={context_docs[-1].get('chunk_index')}"
                )

        full_document_text = self._get_full_document_text(context_docs)
        prompt = f"""
Ты — помощник на основе RAG-системы. Твоя задача - предоставить качественный ответ на вопрос пользователя на основе контекста.

Вопрос пользователя: {original_question}

Контекст:

{full_document_text}

Правила для ответа:
1. Используй ТОЛЬКО информацию из контекста
2. Если в контексте есть прямой ответ, предоставь его кратко и чётко
3. Если ответа нет в контексте или контекст недостаточен, ответь: "В документе нет информации для ответа на этот вопрос."
4. Не придумывай информацию, которой нет в контексте
5. Избегай длинных вводных фраз, переходи сразу к сути

Если вдруг вышло так, что ты не можешь ответить на вопрос на основании предложенного текста, ответь просто: Извините, я не могу ответить на ваш вопрос

Твой ответ:
        """

        for attempt in range(max_retries + 1):
            raw_answer = self.call_ollama(prompt, temperature=temperature, max_tokens=max_tokens)
            answer = raw_answer.strip() if raw_answer else "Не удалось сгенерировать ответ"

            if not validate_answer:
                return answer, True

            # Валидация ответа
            is_valid = self.validate_final_answer(original_question, answer)

            if is_valid or attempt == max_retries:
                if not is_valid and attempt == max_retries:
                    self.logger.warning(f"Ответ не прошел валидацию после {max_retries + 1} попыток")
                return answer, is_valid

            self.logger.info(f"Попытка {attempt + 1}: ответ не прошел валидацию, повторяем генерацию")
            # Немного увеличиваем температуру для разнообразия
            temperature = min(0.7, temperature + 0.1)

        return "Не удалось сгенерировать валидный ответ", False

    def process_question_with_moderation(
        self,
        question: str,
        context_docs: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Полный цикл обработки вопроса с модерацией на входе и выходе.
        """
        self.logger.info("Начинаем обработку вопроса с полной модерацией")

        # Анализ вопроса (включает проверку на неприемлемый контент)
        analysis_result = self.analyze_question(question)

        if analysis_result.get("error"):
            return {
                "success": False,
                "error_type": "content_moderation",
                "error_message": analysis_result["error"],
                "answer": "Извините, я не могу обработать этот вопрос."
            }

        rephrased_question = analysis_result.get("rephrased_question", question)
        if not rephrased_question:
            rephrased_question = question

        # Генерация ответа с валидацией
        answer, is_valid = self.generate_answer(
            rephrased_question,
            context_docs,
            temperature=temperature,
            max_tokens=max_tokens,
            validate_answer=True
        )

        return {
            "success": True,
            "original_question": question,
            "rephrased_question": rephrased_question,
            "answer": answer,
            "is_answer_valid": is_valid,
            "start_date": analysis_result.get("start_date"),
            "end_date": analysis_result.get("end_date")
        }

    def _get_full_document_text(self, chunks: List[Dict]) -> str:
        sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        full_text = " ".join([chunk.get("text", "") for chunk in sorted_chunks])
        self.logger.debug(f"Собран полный текст документа, длина: {len(full_text)} символов")
        return full_text

