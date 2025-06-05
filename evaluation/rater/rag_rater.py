import os
import pandas as pd
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv
import time
import json
import re

class RAGEvaluator:
    """
    Класс для оценки системы RAG.
    
    Параметры:
        api_key (str): Google API key для Gemini
        model: Настроенный экземпляр модели Gemini
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация оценщика с Gemini API.
        
        Аргументы:
            api_key (str, optional): Google API key. Если не предоставлен, будет загружен из .env файла.
            
        Исключения:
            ValueError: Если API key не найден
        """
        load_dotenv()
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set it in .env file or pass it directly.")
        
        # Configure the API with retry settings
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
    def _safe_generate_content(self, prompt: str, max_retries: int = 3) -> str:
        """
        Аргументы:
            prompt (str): Промпт для отправки модели
            max_retries (int): Максимальное количество попыток
            
        Возвращает:
            str: Сгенерированный ответ модели
            
        Исключения:
            Exception: Если все попытки неудачны
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)  
        return ""
        
    def _format_chunks_with_scores(self, chunks: List[str]) -> str:
        formatted_chunks = []
        for chunk in chunks:
            try:
                name_score, content = chunk.split(':', 1)
                name, score = name_score.strip().split(' (')
                score = score.rstrip(')')
                formatted_chunks.append(f"Chunk {name} (Score: {score}):\n{content.strip()}\n")
            except ValueError:
                formatted_chunks.append(f"Chunk:\n{chunk}\n")
        return "\n".join(formatted_chunks)
    
    def evaluate_answer(self, context: str, question: str, model_answer: str) -> Dict[str, Any]:
        """
        Оценить качество ответа модели.
        
        Аргументы:
            context (str): Контекст, предоставленный модели
            question (str): Вопрос, заданный модели
            model_answer (str): Ответ, сгенерированный моделью
            
        Возвращает:
            Dict[str, Any]: Результаты оценки, содержащие score, explanation, и confidence
        """
        prompt = f"""
        You are an expert evaluator of RAG-based chatbot responses. Rate the quality of this answer (0-10) based on:
        - Relevance to question (0-2 points)
        - Factual accuracy compared to context (0-3 points)
        - Completeness of the answer (0-2 points)
        - Coherence and clarity (0-2 points)
        - Proper use of information from the context (0-1 point)

        Context: {context}
        Question: {question}
        Answer: {model_answer}

        Provide your evaluation in the following JSON format:
        {{
            "score": <numerical_score>,
            "explanation": "<detailed_explanation>",
            "confidence": <confidence_score_0_to_1>
        }}

        IMPORTANT: Return ONLY the JSON object, nothing else.
        """
        
        response = self._safe_generate_content(prompt)
        return self._parse_response(response)
    
    def evaluate_chunks(self, question: str, chunks: List[str]) -> Dict[str, Any]:
        """Оценить качество выбранных агентом чанков."""
        formatted_chunks = self._format_chunks_with_scores(chunks)
        
        prompt = f"""
        You are an expert evaluator of RAG chunk selection. Rate the quality of these chunks (0-10) based on:
        - Semantic relevance to question (0-3 points)
        - Coverage of necessary information (0-2 points)
        - Diversity (avoiding redundancy) (0-2 points)
        - Ranking accuracy (similarity scores) (0-2 points)
        - Context completeness (0-1 point)

        Question: {question}
        Chunks with scores:
        {formatted_chunks}

        Provide your evaluation in the following JSON format:
        {{
            "score": <numerical_score>,
            "explanation": "<detailed_explanation>",
            "confidence": <confidence_score_0_to_1>
        }}

        IMPORTANT: Return ONLY the JSON object, nothing else.
        """
        
        response = self._safe_generate_content(prompt)
        return self._parse_response(response)
    
    def evaluate_selected_chunk(self, question: str, selected_chunk: str, all_chunks: List[str]) -> Dict[str, Any]:
        """Оценить качество выбранного агентом главного чанка."""
        prompt = f"""
        You are an expert evaluator of RAG chunk selection. Rate the quality of the selected chunk (0-10) based on:
        - Relevance to question (0-3 points)
        - Sufficiency for answering (0-3 points)
        - Precision (minimal irrelevant info) (0-2 points)
        - Coverage of key points (0-2 points)

        Question: {question}
        Selected chunk: {selected_chunk}
        Available chunks: {all_chunks}

        Provide your evaluation in the following JSON format:
        {{
            "score": <numerical_score>,
            "explanation": "<detailed_explanation>",
            "confidence": <confidence_score_0_to_1>
        }}

        IMPORTANT: Return ONLY the JSON object, nothing else.
        """
        
        response = self._safe_generate_content(prompt)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Парсинг ответ Gemini"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            result = json.loads(response)
            
            required_fields = ['score', 'explanation', 'confidence']
            for field in required_fields:
                if field not in result:
                    result[field] = 0 if field == 'score' or field == 'confidence' else "Missing field"
            
            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {e}")
            print(f"Raw response: {response}")
            return {
                "score": 0,
                "explanation": "Failed to parse response",
                "confidence": 0
            }

def process_csv_file(input_file: str, output_file: str, api_key: str = None, test_mode: bool = False, batch_size: int = 10):
    """
    Обработка CSV файла и оценка всех записей с помощью оценщика.
    
    Аргументы:
        input_file (str): Путь к входному CSV файлу
        output_file (str): Путь к сохранению выходного JSON файла
        api_key (str, optional): API key для Gemini
        test_mode (bool): Для того, чтобы не тратить лишние токены, тестируем одну строку.
        batch_size (int): Количество строк для обработки перед сохранением результатов
        
    Исключения:
        ValueError: Если CSV файл не может быть прочитан с любым поддерживаемым кодировкой
        FileNotFoundError: Если входной файл не существует
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    evaluator = RAGEvaluator(api_key)
    
    # Обработка кодировок для кириллицы
    encodings = ['utf-8', 'cp1251', 'koi8-r', 'iso-8859-5']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Could not read the CSV file with any of the supported encodings")
    
    results = []
    total_rows = 1 if test_mode else len(df)
    
    try:
        for idx, row in tqdm(df.iterrows(), total=total_rows):
            try:
                chunks = [chunk.strip() for chunk in str(row['Top_Chunks']).split('---') if chunk.strip()]

                answer_eval = evaluator.evaluate_answer(
                    str(row['Text']),
                    str(row['Question']),
                    str(row['Answer'])
                )
                
                chunks_eval = evaluator.evaluate_chunks(
                    str(row['Question']),
                    chunks
                )
                
                selected_chunk_eval = evaluator.evaluate_selected_chunk(
                    str(row['Question']),
                    str(row['Selected_Chunk']),
                    chunks
                )
                
                results.append({
                    'question': str(row['Question']),
                    'answer_evaluation': answer_eval,
                    'chunks_evaluation': chunks_eval,
                    'selected_chunk_evaluation': selected_chunk_eval
                })

                if (idx + 1) % batch_size == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"\nSaved intermediate results after processing {idx + 1} rows")

                if test_mode:
                    print("\nTest mode: Processed one row and stopping.")
                    break
                    
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error during processing: {str(e)}")
        if results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        raise
    
    # Final save of all results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing completed. Total rows processed: {len(results)}")
    print(f"Results saved to: {output_file}") 