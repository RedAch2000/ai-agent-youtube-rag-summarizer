from langchain_core.prompts import PromptTemplate



# Summary Prompt
prompt_template_summary = PromptTemplate.from_template("""
You are a professional content summarizer.

Your task is to write a clear, structured summary of a YouTube video based on its transcript.

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
- Extract and explain the main ideas and key points
- Keep the summary between 200–300 words
- Organize the content into logical paragraphs
- Focus only on important insights, conclusions, and messages
- Do not include timestamps, filler speech, or transcript artifacts
- Maintain a neutral, objective, and professional tone
- Write in fluent, natural English

OUTPUT:
Provide a clean, well-structured summary of the video.
""")

# RAG QA Prompt
prompt_rag_qa_template =PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

Question: {question}
Context: {context}
Answer:
""")


