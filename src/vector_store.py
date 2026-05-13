from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class VectorStore:

    def __init__(
        self,
        embedding_model: str,
        persist_directory: str
    ):

        self.embedding = OllamaEmbeddings(
            model=embedding_model
        )

        self.persist_directory = persist_directory

        self.vector_store = None


    def create_vector_store(self, chunks: list[str]):

        documents = [
            Document(page_content=chunk)
            for chunk in chunks
        ]

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )

        return self.vector_store


    def similarity_search(
        self,
        question: str,
        k: int
    ):

        retrieved_context = self.vector_store.similarity_search(
            query=question,
            k=k
        )

        return retrieved_context