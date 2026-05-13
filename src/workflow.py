from .youtube_handler import YoutubeHandler
from .state import AgentState
from langgraph.graph import StateGraph, END, START
from typing import Optional, List
from .text_splitter import TextSplitter 
from .config import Settings
from .vector_store import VectorStore
from langchain_ollama import ChatOllama
from .prompts import prompt_template_summary, prompt_rag_qa_template

class YoutubeRAGWorkflow:
    """This class is used to create a workflow that summarize any video youtube"""
    def __init__(self):
        
        # Loading settings
        self.settings = Settings()
        
        # youtube handler
        self.youtube_handler = YoutubeHandler()
        
        # Text splitter
        self.text_splitter = TextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )

        # vector store
        self.vector_store_manager = VectorStore(
            embedding_model=self.settings.EMBEDDING_MODEL,
            persist_directory=self.settings.PERSIST_DIRECTORY
        )

        # LLM model
        self.llm_model = ChatOllama(
            model=self.settings.LLM_MODEL,
        )
        
        # workflow
        self.workflow = self._create_workflow()


    def process_video(
            self, 
            video_url: str, 
            languages: List[str] = ["en"],
            question: Optional[str] = None
        ):
        """Process Youtube video for QA or summary"""
        
        # Agent State
        state_agent = AgentState(
            video_url=video_url,
            languages=languages
        )

        if question:
            state_agent["question"] = question
        
        # result of workflow
        result = self.workflow.invoke(state_agent)

        # return the result
        return result["response"]
    
    def _create_workflow(self):
        """Create the LangGraph workflow"""
        # initialize graph
        graph = StateGraph(AgentState)

        # nodes of graph
        graph.add_node("extract_video_id", self._extract_video_id)
        graph.add_node("fetch_transcription", self._fetch_transcription)
        graph.add_node("text_splitter", self._text_splitter)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_summary", self._generate_summary)
        graph.add_node("generate_answer", self._generate_answer)

        # edge of graph
        graph.add_edge(START, "extract_video_id")
        graph.add_edge("extract_video_id", "fetch_transcription")

        # Condition Rag_QA or Summary
        graph.add_conditional_edges(
            "fetch_transcription",
            self._control_operation_rag_or_summary,
            {
                "rag_qa": "text_splitter",
                "summary": "generate_summary" 
            }
        )

        # Summary
        graph.add_edge("generate_summary", END)

        # Rag Question/Answer
        graph.add_edge("text_splitter", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_answer")
        graph.add_edge("generate_answer", END)


        # Returning the compiled graph
        return graph.compile()

    def _extract_video_id(self, state: AgentState) -> AgentState:
        """Node that extract the id of the video from the url"""

        video_id = self.youtube_handler.extract_video_id(state["video_url"])        
        # verify if the video id is extracted
        if not video_id:
            raise ValueError("Invalid Youtube URL")
        # video id
        state["video_id"] = video_id
        # print(state["video_id"])
        return state
    

    def _fetch_transcription(self, state: AgentState) -> AgentState:
        """Fetching the transcription of the video"""

        text_transcription = self.youtube_handler.get_video_transcription(
            video_id=state["video_id"],
            languages=state["languages"]
        )
        state["text_transcription"] = text_transcription
        return state
    

    def _generate_summary(self, state: AgentState) -> AgentState:
        """Generate Video Summary"""
        prompt = prompt_template_summary.invoke({
            "transcript": state["text_transcription"]
        })
        summary = self.llm_model.invoke(prompt)
        state["response"] = summary.content
        return state


    def _text_splitter(self, state: AgentState) -> AgentState:
        """ Chunk the transcription text of video """

        text_chunks = self.text_splitter.chunk_text(state["text_transcription"])
        state["chunks"] = text_chunks
        return state

    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """ Retrieve relevant context for question """
        
        vector_store = self.vector_store_manager.create_vector_store(
            chunks=state["chunks"]
        )

        context = vector_store.similarity_search(
            state["question"],
            self.settings.K
        )
        state["context"] = context
        return state


    def _generate_answer(self, state: AgentState) -> AgentState:
        prompt_qa = prompt_rag_qa_template.invoke({
            "context": state["context"],
            "question": state["question"]
        })

        response = self.llm_model.invoke(prompt_qa)
        state["response"] = response.content
        return state



    def _control_operation_rag_or_summary(self, state: AgentState) -> str:
        if "question" in state:
            return "rag_qa"
        return "summary"
        




  
