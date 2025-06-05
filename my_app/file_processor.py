import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        pass

    def read_json_file_sync(self, file_path: str) -> Optional[list]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Ошибка чтения JSON-файла: {e}", exc_info=True)
            return None

    def write_json_file_sync(self, file_path: str, data: list) -> bool:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Ошибка записи JSON-файла: {e}", exc_info=True)
            return False
