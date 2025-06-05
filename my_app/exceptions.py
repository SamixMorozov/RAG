class SearchServiceError(Exception):
    """Базовое исключение для ошибок сервиса поиска"""
    pass

class LLMServiceError(Exception):
    """Ошибки взаимодействия с LLM"""
    pass

class QdrantServiceError(Exception):
    """Ошибки работы с Qdrant"""
    pass

class LLMClientError(Exception):
    """Ошибки работы LLM-клиента"""
    pass
