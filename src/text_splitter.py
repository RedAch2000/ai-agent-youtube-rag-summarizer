from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class TextSplitter():
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk_text(self, text: str) -> List[str]:
        """ Split Text into chunks """
        return self.text_splitter.split_text(text)
        

