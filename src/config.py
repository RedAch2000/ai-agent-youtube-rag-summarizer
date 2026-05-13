from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings.
    """
    
    
    
    # Maximum number of characters in each text chunk
    CHUNK_SIZE: int = 1000

    # Number of overlapping characters between chunks
    # to preserve context continuity
    CHUNK_OVERLAP: int = 200

    PERSIST_DIRECTORY: str = "youtube_video"

    EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"

    # Number of retrieved similar chunks
    K: int = 5

    LLM_MODEL: str = "qwen3.5:4b"

