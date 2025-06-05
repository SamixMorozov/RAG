import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime, timedelta
from IPython import display
from json import loads, dumps
import time
import logging
from urllib.parse import urljoin
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Optional, Tuple
import hashlib

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class rbc_parser:
    def __init__(self, request_delay=1.0, max_retries=3, max_workers=3):
        """
        request_delay: задержка между запросами в секундах
        max_retries: максимальное количество повторных попыток при ошибках
        max_workers: количество потоков для параллельной загрузки
        """
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.session = rq.Session()
        self.lock = threading.Lock()  # Для thread-safe операций

        # Добавляем User-Agent для избежания блокировок
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _make_request(self, url, **kwargs):
        """
        Делает запрос с повторными попытками и обработкой ошибок
        """
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = self.session.get(url, timeout=30, **kwargs)
                response.raise_for_status()
                return response
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1} неудачна для URL {url}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Все попытки исчерпаны для URL {url}")
                    raise e
                time.sleep(2 ** attempt)  # Экспоненциальная задержка

    def _get_url(self, param_dict: dict) -> str:
        """
        Возвращает URL для запроса json таблицы со статьями
        """
        url = 'https://www.rbc.ru/search/ajax/?' + \
              'project={0}&'.format(param_dict['project']) + \
              'category={0}&'.format(param_dict['category']) + \
              'dateFrom={0}&'.format(param_dict['dateFrom']) + \
              'dateTo={0}&'.format(param_dict['dateTo']) + \
              'page={0}&'.format(param_dict['page']) + \
              'query={0}&'.format(param_dict['query']) + \
              'material={0}'.format(param_dict['material'])

        return url

    def _validate_json_file(self, filepath: str) -> bool:
        """
        Проверяет целостность JSON файла в требуемом формате
        """
        try:
            if not os.path.exists(filepath):
                return False

            df = pd.read_json(filepath, orient='records', convert_dates=False)
            if df.empty:
                logger.warning(f"Файл {filepath} пустой")
                return False

            required_cols = ['id', 'title', 'author', 'url', 'text', 'timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"В файле {filepath} отсутствуют столбцы: {missing_cols}")
                return False

            if 'timestamp' in df.columns and len(df) > 0:
                sample_ts = df['timestamp'].dropna().head(1).tolist()
                if sample_ts:
                    ts = sample_ts[0]
                    if not (isinstance(ts, str) and 'T' in ts and len(ts) >= 19):
                        logger.warning(f"Неправильный формат timestamp в файле {filepath}: {ts}")

            logger.info(f"Файл {filepath} валиден, записей: {len(df)}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при проверке файла {filepath}: {e}")
            return False

    def _convert_to_iso_timestamp(self, date_value) -> Optional[str]:
        """
        Конвертирует различные форматы даты в ISO-формат YYYY-MM-DDTHH:MM:SS
        """
        if pd.isna(date_value) or not date_value:
            return None

        try:
            if isinstance(date_value, (int, float)):
                if 1e12 < date_value < 2e12:
                    dt = datetime.fromtimestamp(date_value / 1000)
                elif 1e9 < date_value < 2e9:
                    dt = datetime.fromtimestamp(date_value)
                else:
                    dt = pd.to_datetime(date_value)
            elif isinstance(date_value, str):
                date_clean = date_value
                for tz_marker in ['+', 'Z', 'GMT', 'UTC']:
                    if tz_marker in date_clean:
                        date_clean = date_clean.split(tz_marker)[0].strip()
                dt = pd.to_datetime(date_clean)
            else:
                dt = pd.to_datetime(date_value)
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception as e:
            logger.debug(f"Не удалось преобразовать дату {date_value}: {e}")
            return None

    def _format_data_for_output(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует данные RBC в требуемый формат (ISO timestamp)
        """
        if data.empty:
            return data

        formatted_data = pd.DataFrame()
        formatted_data['id'] = data['id'] if 'id' in data.columns else None
        formatted_data['title'] = data['title'] if 'title' in data.columns else None
        formatted_data['author'] = data.get('project', 'РБК')
        formatted_data['url'] = data['fronturl'] if 'fronturl' in data.columns else None

        text_parts = []
        if 'overview' in data.columns and 'text' in data.columns:
            for _, row in data.iterrows():
                overview = row.get('overview', '') or ''
                text = row.get('text', '') or ''
                if overview and text:
                    full_text = f"{overview}\n\n{text}"
                elif overview:
                    full_text = overview
                elif text:
                    full_text = text
                else:
                    full_text = ""
                text_parts.append(full_text.strip())
        elif 'text' in data.columns:
            text_parts = data['text'].fillna('').tolist()
        elif 'body' in data.columns:
            text_parts = data['body'].fillna('').tolist()
        else:
            text_parts = [''] * len(data)

        formatted_data['text'] = text_parts

        timestamps = []
        date_field = None
        for field in ['publish_date', 'publish_date_t', 'date', 'datetime', 'published_at']:
            if field in data.columns:
                date_field = field
                break

        if date_field:
            for date_value in data[date_field]:
                iso_timestamp = self._convert_to_iso_timestamp(date_value)
                timestamps.append(iso_timestamp)
        else:
            logger.warning("Поле с датой не найдено, используется текущее время")
            current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            timestamps = [current_time] * len(data)

        formatted_data['timestamp'] = timestamps
        formatted_data = formatted_data.dropna(subset=['id', 'title'])

        if len(formatted_data) > 0:
            sample_timestamps = formatted_data['timestamp'].dropna().head(3).tolist()
            if sample_timestamps:
                logger.info(f"Примеры timestamp в ISO-формате: {sample_timestamps}")

        return formatted_data

    def _save_chunk_json(self, data: pd.DataFrame, chunk_name: str, output_dir: str) -> str:
        """
        Сохраняет кусок данных в JSON с проверкой целостности и правильным форматированием
        """
        if data.empty:
            return None

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{chunk_name}.json")

        try:
            formatted_data = self._format_data_for_output(data)
            if formatted_data.empty:
                logger.warning("Нет данных после форматирования")
                return None

            data_cleaned = formatted_data.drop_duplicates(subset=['id'])
            logger.info(f"Удалено {len(formatted_data) - len(data_cleaned)} дубликатов")

            data_cleaned.to_json(
                filepath,
                orient='records',
                force_ascii=False,
                indent=2,
                date_format=None,
                default_handler=str
            )

            if self._validate_json_file(filepath):
                logger.info(f"Кусок сохранен: {filepath}, записей: {len(data_cleaned)}")
                return filepath
            else:
                logger.error(f"Файл {filepath} поврежден после сохранения")
                return None

        except Exception as e:
            logger.error(f"Ошибка при сохранении куска {filepath}: {e}")
            return None

    def _get_article_data_batch(self, urls: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Параллельная загрузка данных статей
        """
        results = [None] * len(urls)

        def fetch_article(idx_url):
            idx, url = idx_url
            return idx, self._get_article_data(url)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(fetch_article, (i, url)): i for i, url in enumerate(urls)}
            for future in as_completed(future_to_idx):
                try:
                    idx, (overview, text) = future.result()
                    results[idx] = (overview, text)
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.warning(f"Ошибка при загрузке статьи {idx}: {e}")
                    results[idx] = (None, None)
        return results

    def _get_search_table(self, param_dict: dict, include_text: bool = True) -> pd.DataFrame:
        """
        Возвращает pd.DataFrame со списком статей
        """
        try:
            url = self._get_url(param_dict)
            logger.info(f"Запрос к URL: {url}")
            r = self._make_request(url)

            try:
                json_data = r.json()
            except ValueError as e:
                logger.error(f"Ошибка парсинга JSON: {e}")
                return pd.DataFrame()

            if 'items' not in json_data:
                logger.warning("Ключ 'items' не найден в ответе")
                return pd.DataFrame()

            if not json_data['items']:
                logger.info("Пустой список статей")
                return pd.DataFrame()

            search_table = pd.DataFrame(json_data['items'])
            logger.info(f"Найдено {len(search_table)} статей")

            if include_text and not search_table.empty:
                logger.info("Загрузка текста статей (параллельно)...")
                urls = search_table['fronturl'].tolist()
                texts = self._get_article_data_batch(urls)
                search_table[['overview', 'text']] = texts

            if 'publish_date_t' in search_table.columns:
                search_table = search_table.sort_values('publish_date_t', ignore_index=True)

            return search_table

        except Exception as e:
            logger.error(f"Ошибка в _get_search_table: {e}")
            return pd.DataFrame()

    def _iterable_load_by_page(self, param_dict):
        """
        Загружает все страницы
        """
        param_copy = param_dict.copy()
        results = []
        page_num = 0
        empty_pages_count = 0
        max_empty_pages = 3

        while empty_pages_count < max_empty_pages:
            param_copy['page'] = str(page_num)
            logger.info(f"Загрузка страницы {page_num}")
            result = self._get_search_table(param_copy, include_text=False)

            if result.empty:
                empty_pages_count += 1
                logger.info(f"Пустая страница {page_num}, счетчик пустых страниц: {empty_pages_count}")
            else:
                empty_pages_count = 0
                results.append(result)
                logger.info(f"Страница {page_num}: найдено {len(result)} статей")

            page_num += 1
            if page_num > 1000:
                logger.warning("Достигнут лимит страниц (1000), остановка")
                break

        if results:
            combined_results = pd.concat(results, axis=0, ignore_index=True)
            logger.info(f"Всего найдено {len(combined_results)} статей")
            return combined_results
        else:
            logger.warning("Не найдено ни одной статьи")
            return pd.DataFrame()

    def _get_article_data(self, url: str):
        """
        Возвращает описание и текст статьи по ссылке
        """
        try:
            if not url or not url.startswith('http'):
                if url:
                    url = urljoin('https://www.rbc.ru', url)
                else:
                    return None, None

            r = self._make_request(url)
            soup = bs(r.text, features="lxml")

            overview = None
            div_overview = soup.find('div', {'class': 'article__text__overview'})
            if div_overview:
                overview = div_overview.get_text().replace('<br />', '\n').strip()

            if not overview:
                meta_desc = soup.find('meta', {'name': 'description'})
                if meta_desc:
                    overview = meta_desc.get('content', '').strip()

            text = None
            article_selectors = [
                'div.article__text p',
                'div.article__content p',
                '.js-mediator-article p',
                'p'
            ]

            for selector in article_selectors:
                paragraphs = soup.select(selector)
                if paragraphs:
                    text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    if len(text) > 100:
                        break

            return overview, text

        except Exception as e:
            logger.warning(f"Ошибка при парсинге статьи {url}: {e}")
            return None, None

    def _find_latest_checkpoint(self, output_dir: str, base_name: str) -> Optional[str]:
        """
        Находит последний чекпоинт
        """
        pattern = os.path.join(output_dir, f"{base_name}_*.json")
        checkpoints = glob.glob(pattern)

        if not checkpoints:
            return None

        valid_checkpoints = []
        for cp in checkpoints:
            if self._validate_json_file(cp):
                valid_checkpoints.append(cp)
            else:
                logger.warning(f"Поврежденный чекпоинт удален: {cp}")
                try:
                    os.remove(cp)
                except:
                    pass

        if not valid_checkpoints:
            return None

        latest_date = None
        latest_file = None
        for cp in valid_checkpoints:
            try:
                parts = os.path.basename(cp).split('_')
                if len(parts) >= 4:
                    date_str = parts[-1].replace('.json', '')
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    if latest_date is None or date_obj > latest_date:
                        latest_date = date_obj
                        latest_file = cp
            except:
                continue

        if latest_file:
            logger.info(f"Найден последний чекпоинт: {latest_file}")

        return latest_file

    def _merge_json_chunks(self, output_dir: str, base_name: str, final_filename: str) -> pd.DataFrame:
        """
        Объединяет все JSON куски
        """
        pattern = os.path.join(output_dir, f"{base_name}_*.json")
        chunk_files = glob.glob(pattern)

        if not chunk_files:
            logger.warning("Не найдено кусков для объединения")
            return pd.DataFrame()

        logger.info(f"Найдено {len(chunk_files)} кусков для объединения")

        all_data = []
        valid_files = []

        for chunk_file in sorted(chunk_files):
            if self._validate_json_file(chunk_file):
                try:
                    df = pd.read_json(chunk_file, orient='records', convert_dates=False)

                    if 'timestamp' in df.columns:
                        fixed_timestamps = []
                        for ts in df['timestamp']:
                            if pd.isna(ts) or not ts:
                                fixed_timestamps.append(None)
                            elif isinstance(ts, str) and 'T' in ts and len(ts) == 19:
                                fixed_timestamps.append(ts)
                            else:
                                iso_ts = self._convert_to_iso_timestamp(ts)
                                fixed_timestamps.append(iso_ts)
                        df['timestamp'] = fixed_timestamps

                    all_data.append(df)
                    valid_files.append(chunk_file)
                    logger.info(f"Загружен кусок: {chunk_file}, записей: {len(df)}")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке куска {chunk_file}: {e}")

        if not all_data:
            logger.error("Не удалось загрузить ни одного валидного куска")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, axis=0, ignore_index=True)

        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['id'])
        final_count = len(combined_df)

        logger.info(f"Объединено данных: {initial_count}, после удаления дубликатов: {final_count}")

        if 'timestamp' in combined_df.columns:
            sample_timestamps = combined_df['timestamp'].dropna().head(5).tolist()
            logger.info(f"Примеры timestamp после объединения: {sample_timestamps}")

        try:
            combined_df.to_json(
                final_filename,
                orient='records',
                force_ascii=False,
                indent=2,
                date_format=None
            )
            logger.info(f"Финальный файл сохранен: {final_filename}")

            test_df = pd.read_json(final_filename, orient='records', convert_dates=False)
            if 'timestamp' in test_df.columns:
                test_timestamps = test_df['timestamp'].dropna().head(3).tolist()
                logger.info(f"Проверка: timestamp в сохраненном файле: {test_timestamps}")

            backup_dir = os.path.join(output_dir, 'chunks_backup')
            os.makedirs(backup_dir, exist_ok=True)
            for chunk_file in valid_files:
                backup_path = os.path.join(backup_dir, os.path.basename(chunk_file))
                os.rename(chunk_file, backup_path)

            logger.info(f"Куски перемещены в бэкап: {backup_dir}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении финального файла: {e}")

        return combined_df

    def fix_json_timestamps(self, json_file_path: str) -> bool:
        """
        Исправляет timestamp в существующем JSON файле
        """
        try:
            logger.info(f"Исправление timestamp в файле: {json_file_path}")

            df = pd.read_json(json_file_path, orient='records', convert_dates=False)

            if 'timestamp' not in df.columns:
                logger.warning("Столбец timestamp не найден")
                return False

            fixed_timestamps = []
            for ts in df['timestamp']:
                iso_ts = self._convert_to_iso_timestamp(ts)
                fixed_timestamps.append(iso_ts)

            df['timestamp'] = fixed_timestamps

            backup_path = json_file_path.replace('.json', '_backup.json')
            os.rename(json_file_path, backup_path)
            logger.info(f"Создан бэкап: {backup_path}")

            df.to_json(
                json_file_path,
                orient='records',
                force_ascii=False,
                indent=2,
                date_format=None
            )

            test_df = pd.read_json(json_file_path, orient='records', convert_dates=False)
            sample_timestamps = test_df['timestamp'].dropna().head(5).tolist()
            logger.info(f"Timestamp после исправления: {sample_timestamps}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при исправлении файла: {e}")
            return False

    def fix_all_json_timestamps_in_directory(self, directory: str) -> None:
        """
        Исправляет timestamp во всех JSON файлах в указанной директории
        """
        json_files = glob.glob(os.path.join(directory, "*.json"))

        logger.info(f"Найдено {len(json_files)} JSON файлов для проверки")

        for json_file in json_files:
            try:
                if '_backup' in json_file:
                    continue

                logger.info(f"Проверка файла: {json_file}")

                df = pd.read_json(json_file, orient='records', convert_dates=False)

                if 'timestamp' not in df.columns:
                    logger.warning(f"В файле {json_file} нет столбца timestamp")
                    continue

                needs_fix = False
                sample_ts = df['timestamp'].dropna().head(1).tolist()
                if sample_ts:
                    ts = sample_ts[0]
                    if not (isinstance(ts, str) and 'T' in ts and len(ts) == 19):
                        needs_fix = True

                if needs_fix:
                    logger.info(f"Файл {json_file} требует исправления timestamp")
                    self.fix_json_timestamps(json_file)
                else:
                    logger.info(f"Файл {json_file} имеет правильный формат timestamp")

            except Exception as e:
                logger.error(f"Ошибка при обработке файла {json_file}: {e}")

    def validate_final_json(self, json_file_path: str) -> Dict[str, any]:
        """
        Проверяет финальный JSON файл и возвращает статистику
        """
        try:
            df = pd.read_json(json_file_path, orient='records', convert_dates=False)

            stats = {
                'total_records': len(df),
                'valid_timestamps': 0,
                'invalid_timestamps': 0,
                'missing_timestamps': 0,
                'sample_timestamps': [],
                'issues': []
            }

            if 'timestamp' in df.columns:
                for ts in df['timestamp']:
                    if pd.isna(ts) or not ts:
                        stats['missing_timestamps'] += 1
                    elif isinstance(ts, str) and 'T' in ts and len(ts) == 19:
                        stats['valid_timestamps'] += 1
                    else:
                        stats['invalid_timestamps'] += 1

                stats['sample_timestamps'] = df['timestamp'].dropna().head(5).tolist()
            else:
                stats['issues'].append('Отсутствует столбец timestamp')

            required_fields = ['id', 'title', 'author', 'url', 'text']
            for field in required_fields:
                if field not in df.columns:
                    stats['issues'].append(f'Отсутствует столбец {field}')
                else:
                    null_count = df[field].isna().sum()
                    if null_count > 0:
                        stats['issues'].append(f'Поле {field} содержит {null_count} пустых значений')

            if 'id' in df.columns:
                duplicates = df.duplicated(subset=['id']).sum()
                if duplicates > 0:
                    stats['issues'].append(f'Найдено {duplicates} дубликатов по id')

            return stats

        except Exception as e:
            return {'error': str(e)}

    def get_articles(self,
                     param_dict,
                     time_step=1,
                     save_every=3,
                     output_dir="./rbc_data",
                     resume_from_checkpoint=True) -> pd.DataFrame:
        """
        Функция для скачивания статей
        """
        param_copy = param_dict.copy()
        time_step_delta = timedelta(days=time_step)
        dateFrom = datetime.strptime(param_copy['dateFrom'], '%d.%m.%Y')
        dateTo = datetime.strptime(param_copy['dateTo'], '%d.%m.%Y')

        if dateFrom > dateTo:
            raise ValueError('dateFrom should be less than dateTo')

        os.makedirs(output_dir, exist_ok=True)
        base_name = f"rbc_{param_dict['dateFrom'].replace('.', '-')}_{param_dict['dateTo'].replace('.', '-')}"

        current_date = dateFrom
        if resume_from_checkpoint:
            latest_checkpoint = self._find_latest_checkpoint(output_dir, base_name.split('_')[0])
            if latest_checkpoint:
                try:
                    parts = os.path.basename(latest_checkpoint).split('_')
                    if len(parts) >= 4:
                        last_date_str = parts[-1].replace('.json', '')
                        last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                        current_date = last_date + timedelta(days=1)
                        logger.info(f"Продолжаем с даты: {current_date.strftime('%d.%m.%Y')}")
                except Exception as e:
                    logger.warning(f"Не удалось определить дату из чекпоинта: {e}")

        save_counter = 0
        while current_date <= dateTo:
            end_date = min(current_date + time_step_delta, dateTo)
            param_copy['dateFrom'] = current_date.strftime("%d.%m.%Y")
            param_copy['dateTo'] = end_date.strftime("%d.%m.%Y")

            logger.info(f'Парсинг статей с {param_copy["dateFrom"]} по {param_copy["dateTo"]}')

            try:
                daily_data = self._iterable_load_by_page(param_copy)

                if not daily_data.empty:
                    if 'text' not in daily_data.columns:
                        logger.info("Загрузка текста статей (параллельно)...")
                        urls = daily_data['fronturl'].tolist()
                        texts = self._get_article_data_batch(urls)
                        daily_data[['overview', 'text']] = texts

                    chunk_name = f"{base_name.split('_')[0]}_{current_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
                    saved_path = self._save_chunk_json(daily_data, chunk_name, output_dir)

                    if saved_path:
                        logger.info(f"Кусок сохранен: {saved_path}")
                    else:
                        logger.error("Ошибка при сохранении куска")
                else:
                    logger.info("Статей не найдено за данный период")

            except Exception as e:
                logger.error(f"Ошибка при парсинге периода {param_copy['dateFrom']} - {param_copy['dateTo']}: {e}")

            current_date = end_date + timedelta(days=1)
            save_counter += 1

        final_filename = os.path.join(output_dir, f"{base_name}.json")
        final_data = self._merge_json_chunks(output_dir, base_name.split('_')[0], final_filename)

        logger.info(f'Завершено. Всего собрано {len(final_data)} статей')
        return final_data


def main():
    """
    Главная функция с поддержкой аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='RBC News Parser')
    parser.add_argument('--date-from', help='Дата начала (YYYY-MM-DD)')
    parser.add_argument('--date-to', help='Дата окончания (YYYY-MM-DD)')
    parser.add_argument('--query', default='РБК', help='Поисковый запрос')
    parser.add_argument('--project', default='rbcnews', help='Проект RBC')
    parser.add_argument('--category', default='TopRbcRu_economics', help='Категория')
    parser.add_argument('--output-dir', default='./rbc_data', help='Директория для сохранения')
    parser.add_argument('--time-step', type=int, default=1, help='Шаг по дням')
    parser.add_argument('--max-workers', type=int, default=3, help='Количество потоков')
    parser.add_argument('--request-delay', type=float, default=1.5, help='Задержка между запросами')
    parser.add_argument('--no-resume', action='store_true', help='Не продолжать с чекпоинта')
    parser.add_argument('--fix-timestamps', action='store_true', help='Исправить timestamp в существующих файлах')
    parser.add_argument('--fix-file', help='Путь к конкретному файлу для исправления timestamp')
    parser.add_argument('--validate-file', help='Путь к файлу для валидации')

    args = parser.parse_args()

    rbc_parser_instance = rbc_parser(
        request_delay=args.request_delay,
        max_retries=3,
        max_workers=args.max_workers
    )

    if args.fix_timestamps:
        rbc_parser_instance.fix_all_json_timestamps_in_directory(args.output_dir)
        return

    if args.fix_file:
        success = rbc_parser_instance.fix_json_timestamps(args.fix_file)
        if success:
            logger.info("Файл успешно исправлен")
        else:
            logger.error("Ошибка при исправлении файла")
        return

    if args.validate_file:
        stats = rbc_parser_instance.validate_final_json(args.validate_file)
        print("\nСтатистика файла:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        return

    if not args.date_from or not args.date_to:
        parser.error("--date-from и --date-to обязательны для режима парсинга")

    param_dict = {
        'query': args.query,
        'project': args.project,
        'category': args.category,
        'dateFrom': datetime.strptime(args.date_from, '%Y-%m-%d').strftime('%d.%m.%Y'),
        'dateTo': datetime.strptime(args.date_to, '%Y-%m-%d').strftime('%d.%m.%Y'),
        'page': '0',
        'material': ""
    }

    result = rbc_parser_instance.get_articles(
        param_dict=param_dict,
        time_step=args.time_step,
        output_dir=args.output_dir,
        resume_from_checkpoint=not args.no_resume
    )

    logger.info(f"Парсинг завершен. Собрано {len(result)} статей")

    final_file = os.path.join(args.output_dir, f"rbc_{args.date_from}_{args.date_to}.json")
    if os.path.exists(final_file):
        stats = rbc_parser_instance.validate_final_json(final_file)
        logger.info(f"Валидация финального файла: {stats}")


if __name__ == "__main__":
    main()
