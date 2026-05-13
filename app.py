from src.workflow import YoutubeRAGWorkflow
from src.state import AgentState

def main():
    youtube_rag_workflow = YoutubeRAGWorkflow()

    result = youtube_rag_workflow.process_video(
        video_url= "https://www.youtube.com/watch?v=VSFuqMh4hus",
        languages=["en"],
        # question="what is vector database?"
    )

    print(result)

if __name__ == "__main__":
    main()