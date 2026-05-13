from typing import TypedDict, List, Any

class AgentState(TypedDict):
    """State for Youtube RAG Workflow"""
    question: str = None
    video_url: str =None
    video_id: str = None
    text_transcription: str = None
    languages: List[str] = None
    chunks: List[str] = None
    vector_store: Any = None
    context: Any = None
    response: str = None
    
