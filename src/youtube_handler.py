from typing import List, Dict, Optional
from pydantic import BaseModel
import re
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
from youtube_transcript_api.formatters import TextFormatter

class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float


class YoutubeHandler:
    """This class is a Handler for Youtube video transcript operations"""

    
    def extract_video_id(
        self, 
        video_url: str
    ) -> Optional[str]:
        """ Extract video ID from Youtube URL """
        
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
            
        return None
    
    def get_video_transcription(
        self,
        video_id: str,
        languages: List[str] = ["en"]
    ) -> str:
        """This function gets the transcription text of a video."""

        try:
            # Fetch transcript
            ytt_api = YouTubeTranscriptApi()

            transcription_data = ytt_api.fetch(
                video_id=video_id,
                languages=languages
            )

            # Convert transcript to text
            formatter = TextFormatter()

            transcription_text = formatter.format_transcript(
                transcription_data
            )

            return transcription_text

        except NoTranscriptFound:
            raise Exception(
                f"No transcript found for video: {video_id}"
            )

        except TranscriptsDisabled:
            raise Exception(
                f"Transcripts are disabled for video: {video_id}"
            )

        except VideoUnavailable:
            raise Exception(
                f"Video unavailable: {video_id}"
            )

        except Exception as e:
            raise Exception(
                f"Error while fetching transcript: {str(e)}"
            )
            
      
   

    
