from typing_extensions import Annotated, List, TypedDict, Literal, Optional, Dict, Any, Union, NotRequired
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ReportStateInput(TypedDict):
    topic: str # Report topic
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search

class ReportState(TypedDict):
    topic: str # Report topic    
    feedback_on_report_plan: Annotated[list[str], operator.add] # List of feedback on the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: Annotated[str, operator.add] # String of formatted source content from web search

class SectionState(TypedDict):
    topic: str # Report topic
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search


# # Extending.... (for my purpose)

class DialogueLine(TypedDict):
    speaker: Literal['stewie', 'peter']
    speaker_img: Literal[
        'stewie.png',

        'peter.png'
    ]
    search_query: Annotated[Optional[str], 'search query -> string data type or None data type (Not even empty string)']
    text: str


class ScriptOutputState(TypedDict):
    dialogues: List[DialogueLine]
    topic: Annotated[str, 'Topic under 5 words']


# # Image
class ImageSelected(TypedDict):
    description: str
    uri: str

class TavilyImageResult(TypedDict):
    url: str
    description: Optional[str]

class DialogueLineAudio(DialogueLine):
    image_uri: Optional[str]
    image_download_path: Optional[str]

    mov_path: Optional[str]

    audio_name: str
    file_path: str
    start: float
    end: float
    duration: float


class ReelMetaData(TypedDict):
    """Structured output for Instagram Reel metadata."""
    title: str
    description: str
    catchy_text: str

class BestImageIndex(TypedDict):
    index: str

class ScriptState(ReportStateOutput, ScriptOutputState):
    script_output_dir_path: str
    audio_path_base: str

    audio_dialogues: List[DialogueLineAudio]
    combined_audio_path: str
    complete_duration: float

    
    video_path: str

    subtitles_path: str

    captioned_video_path: str

    video_metadata: ReelMetaData

    thumbnail_path: str

    final_state_json_path: str