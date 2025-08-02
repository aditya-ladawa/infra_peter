from typing_extensions import Annotated, List, TypedDict, Literal, Optional, Dict, Any, Union, NotRequired

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback,

    # # my classes
    DialogueLine,
    ScriptOutputState,
    ScriptState,

    ImageSelected,
    TavilyImageResult,

    DialogueLineAudio,

    ReelMetaData,

    BestImageIndex,
    

)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str
)

# # Extension imports
from langchain_tavily import TavilySearch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from open_deep_research.handle_google_drive import (
    authenticate_google_drive,
    create_folder,
    download_file,
    update_file,
    delete_file,
    upload_file,
    find_folder_id_by_name,
    list_files,
)
from open_deep_research.anim import Animations
from open_deep_research.handle_captions import VideoCaptioner

from functools import partial


import os
import re
import json
import math
import random
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
import tempfile
from collections import defaultdict
from io import BytesIO

import uuid
import aiofiles

import aiohttp
import numpy as np
import ffmpeg
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
from collections import Counter
from moviepy import VideoFileClip, CompositeVideoClip, TextClip, ImageClip

import openpyxl
import csv


from collections import Counter, defaultdict
from pilmoji import Pilmoji
from pilmoji.source import GoogleEmojiSource


from faster_whisper import WhisperModel


## Nodes -- 
SCREEN_COVERAGE = 0.5


async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """

    # Inputs
    topic = state["topic"]

    # Get list of feedback on the report plan
    feedback_list = state.get("feedback_on_report_plan", [])

    # Concatenate feedback on the report plan into a single string
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs, temperature=0) 

    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today=get_today_str()
    )

    # Generate queries  
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

    else:
        # With other models, thinking tokens are not specifically allocated
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider,
                                      model_kwargs=planner_model_kwargs)
    
    # Generate the report sections
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps."""

    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)
    # Handle dict type returned from interrupt
    if isinstance(feedback, dict):
        # Extract the boolean or string value from the dict
        feedback_value = next(iter(feedback.values()))
        if feedback_value is True:
            return Command(goto=[
                Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0})
                for s in sections if s.research
            ])
        elif isinstance(feedback_value, str):
            return Command(goto="generate_report_plan", update={"feedback_on_report_plan": [feedback_value]})
        else:
            raise TypeError(f"Interrupt dict feedback value type {type(feedback_value)} is not supported.")
    
    if isinstance(feedback, bool) and feedback is True:
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections if s.research
        ])
    
    if isinstance(feedback, str):
        return Command(goto="generate_report_plan", update={"feedback_on_report_plan": [feedback]})
    
    raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")

async def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries,
                                                           today=get_today_str())

    # Generate queries  
    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs, temperature=0) 

    section_content = await writer_model.ainvoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, model_kwargs=planner_model_kwargs).with_structured_output(Feedback)
    # Generate feedback
    feedback = await reflection_model.ainvoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        update = {"completed_sections": [section]}
        if configurable.include_source_str:
            update["source_str"] = source_str
        return Command(update=update, goto=END)

    # Update the existing section with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web"
        )
    
async def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections, temperature=0)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState, config: RunnableConfig):
    """Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    if configurable.include_source_str:
        return {"final_report": all_sections, "source_str": state["source_str"]}
    else:
        return {"final_report": all_sections}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]




# My vars
files_dir = os.path.join('src/open_deep_research/files')


# # Extending nodes
def sanitize_topic_title(topic: str) -> str:
    """
    Sanitizes a topic title by:
    - Removing special characters
    - Replacing spaces with underscores
    - Converting to lowercase

    Example:
        "Redis: Speed & Performance!" → "redis_speed_performance"
    """
    cleaned = re.sub(r'[^a-zA-Z0-9 ]+', '', topic)  # Remove non-alphanumeric and non-space
    return cleaned.strip().lower().replace(" ", "_")

async def script_generator(state: ScriptState, config: RunnableConfig) -> ScriptState:
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs, temperature=0.3) 
    # writer_model = init_chat_model(model='deepseek-chat', model_provider='deepseek', model_kwargs=writer_model_kwargs, temperature=0.15) 

    structured_llm = writer_model.with_structured_output(Queries)

    # Prompts
    system_template = configurable.script_gen_system_prompt
    human_template = """
    Important: Rewrite the provided topic to a 4-word max title.

    Generate a reel script on the topic below:\n\n{topic}\n\n
    Use this detailed report as your source:\n\n{final_report}
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),

    ])

    formatted_messages = chat_prompt.format_messages(
        topic=state["topic"],
        final_report=state["final_report"]
    )

    script = await writer_model.with_structured_output(ScriptOutputState).ainvoke(formatted_messages)

    sanitized_topic = sanitize_topic_title(script['topic'])
    return {"topic": sanitized_topic, "dialogues": script['dialogues']}




async def push_for_cloning(state: ScriptState) -> ScriptState:

    raw_dialogue = state['dialogues']
    topic = state['topic']
    filename = f"raw_dialogue_{topic}.json"

    # Local file path
    local_dir = os.path.join(files_dir, topic)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    # Save to known path
    with open(local_path, 'w') as f:
        json.dump(raw_dialogue, f, indent=2)

    # Google Drive operations
    service = authenticate_google_drive()

    root_id = 'root'
    channel_automations_folder = find_folder_id_by_name(service, 'channel_automations', root_id)
    if not channel_automations_folder:
        raise RuntimeError("Missing 'channel_automations' folder")

    peterai_folder = find_folder_id_by_name(service, 'peterAI', channel_automations_folder)
    if not peterai_folder:
        raise RuntimeError("Missing 'peterAI' folder")

    scripts_folder = create_folder(service, 'SCRIPTS', peterai_folder)
    script_topic_folder = create_folder(service, topic, scripts_folder)

    # Check if file exists
    existing_files = list_files(service, script_topic_folder) or []
    for file in existing_files:
        if file['name'] == filename:
            print(f"'{filename}' already exists. Deleting old version.")
            delete_file(service, file['id'])
            break

    # Upload the new file
    upload_file(service, local_path, script_topic_folder)

    return {'script_output_dir_path': local_dir}



def format_image_options(images: List[Dict[str, Any]]) -> str:
    return "\n".join([
        f"- [{i}] {img['description'] or 'No description'}\n  URL: {img['url']}" for i, img in enumerate(images)
    ])

async def tavily_image_search(query: str) -> List[Dict[str, Any]]:
    search_tool = TavilySearch(max_results=5, include_images=True, include_image_descriptions=True)

    raw_results = await search_tool.ainvoke({'query': query})

    # Fix: decode stringified JSON if needed
    if isinstance(raw_results, str):
        try:
            results = json.loads(raw_results)
        except json.JSONDecodeError as e:
            print(f"Failed to parse Tavily result: {e}")
            return []
    else:
        results = raw_results

    return results.get('images', [])


# Helper: Resize non-GIF images
def resize_image_exact(image_path: str, output_path: str, width=1080, height=864):
    with Image.open(image_path) as img:
        img = img.convert("RGBA")  # Ensure transparency is preserved

        resized_img = img.resize((width, height), Image.LANCZOS)

        # Add white background to eliminate transparency
        white_bg = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        white_bg.paste(resized_img, (0, 0), resized_img)

        # Convert to RGB before saving to JPEG (JPEG doesn't support alpha)
        white_bg.convert("RGB").save(output_path, format="JPEG")


# Helper: Detect GIF via extension
def is_gif(path: Path) -> bool:
    return path.suffix.lower() == ".gif"


def add_white_background(img: Image.Image, padding=0) -> Image.Image:
    w, h = img.size
    bg_w = w + 2*padding
    bg_h = h + 2*padding
    white_bg = Image.new("RGBA", (bg_w, bg_h), (255, 255, 255, 255))
    white_bg.paste(img, (padding, padding), img)
    return white_bg

async def decide_images(state: ScriptState, config: RunnableConfig) -> ScriptState:
    configurable = WorkflowConfiguration.from_runnable_config(config)

    audio_dialogue = state["dialogues"]
    sanitized_title = state['topic']

    illustrations_dir = os.path.join(files_dir, sanitized_title, 'illustrations')
    os.makedirs(illustrations_dir, exist_ok=True)

    image_select_system_prompt = configurable.image_decide_prompt

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    # writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs, temperature=0)
    writer_model = init_chat_model(model='deepseek-chat', model_provider='deepseek', model_kwargs=writer_model_kwargs, temperature=0.15) 

    async def download_image(url: str, dest_path: Path):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(dest_path, "wb") as f:
                        f.write(content)
                else:
                    raise RuntimeError(f"Failed to download image {url}, status {resp.status}")

    async def try_download_and_resize(uri: str, dest_path: Path) -> str | None:
        try:
            await download_image(uri, dest_path)
            if is_gif(dest_path):
                return str(dest_path)  # No resizing for GIFs
            # Resize and save as JPEG
            normalized_path = dest_path.with_stem(dest_path.stem + "_normalized").with_suffix(".jpg")
            resize_image_exact(dest_path, normalized_path)
            return str(normalized_path)
        except Exception as e:
            print(f"Download or resize failed for {uri}: {e}")
            return None

    async def process_line(i: int, line: DialogueLineAudio) -> DialogueLineAudio:
        if not line.get("search_query"):
            return line

        results = await tavily_image_search(line["search_query"])
        if not results:
            return line

        prompt = image_select_system_prompt + f"""
            Dialogue line:
            "{line['text']}"

            Speaker: {line['speaker']}

            Image options:
            {format_image_options(results)}
        """

        image: ImageSelected = await writer_model.with_structured_output(ImageSelected).ainvoke(prompt)

        if image["uri"]:
            filename = os.path.basename(image["uri"].split("?")[0])
            dest_path = Path(os.path.join(illustrations_dir, filename))

            normalized_path = await try_download_and_resize(image["uri"], dest_path)
            if normalized_path:
                line["image_uri"] = image["uri"]
                line["image_download_path"] = normalized_path
            else:
                retry_prompt = f"""
                The image download failed for this search query: "{line['search_query']}"
                Please provide a refined or alternative search query to find a suitable image for the following dialogue line:
                "{line['text']}"
                Speaker: {line['speaker']}

                IMPORTANT: STRICTLY OUTPUT THE SEARCH QUERY STRING. NOTHING ELSE
                """
                new_search_query = await writer_model.ainvoke(retry_prompt)
                new_search_query = new_search_query.content.strip()

                if new_search_query:
                    new_results = await tavily_image_search(new_search_query)
                    if new_results:
                        new_image_url = new_results[0]['url']
                        new_filename = os.path.basename(new_image_url.split("?")[0])
                        new_dest_path = Path(os.path.join(illustrations_dir, new_filename))
                        normalized_path = await try_download_and_resize(new_image_url, new_dest_path)
                        if normalized_path:
                            line["image_uri"] = new_image_url
                            line["image_download_path"] = normalized_path
                else:
                    print("LLM did not return a new search query")

        return line

    updated_audio_dialogue = await asyncio.gather(*[
        process_line(i, line) for i, line in enumerate(audio_dialogue)
    ])

    return {
        "audio_dialogues": updated_audio_dialogue
    }




async def create_movs(state: ScriptState) -> ScriptState:
    dialogues = state["audio_dialogues"]
    script_path = state["script_output_dir_path"]
    SFX_DIR = "src/open_deep_research/RESOURCES/SFX"

    MOV_DIR = os.path.join(script_path, "fx_movs")
    os.makedirs(MOV_DIR, exist_ok=True)

    anim = Animations(output_path=MOV_DIR, illustration_height_percent=SCREEN_COVERAGE, padding_percent=0.05)

    fx_defs = [
        {"animation": "bubble_pop",              "duration": 0.15, "sfx": "pop1",        "function": anim.bubble_pop},
        {"animation": "slide_in_left",           "duration": 0.3,  "sfx": "swish",       "function": partial(anim.slide_in, direction="left")},
        {"animation": "slide_in_right",          "duration": 0.3,  "sfx": "swish",       "function": partial(anim.slide_in, direction="right")},
        {"animation": "slide_in_overshoot_left", "duration": 0.5,  "sfx": "swoosh",      "function": partial(anim.slide_in_overshoot, direction="left")},
        {"animation": "slide_in_overshoot_right","duration": 0.5,  "sfx": "swoosh",      "function": partial(anim.slide_in_overshoot, direction="right")},
        {"animation": "fade_in",                 "duration": 0.5,  "sfx": "darkwoosh",   "function": anim.fade_in},
        {"animation": "blur_in",                 "duration": 0.5,  "sfx": "shimmer",     "function": anim.blur_in},
        {"animation": "drop",                    "duration": 0.5,  "sfx": "thud",        "function": anim.drop},
        {"animation": "shake",                   "duration": 0.3,  "sfx": "chime",       "function": anim.shake},
        {"animation": "bounce",                  "duration": 0.3,  "sfx": "bpop2",       "function": anim.bounce},
        {"animation": "blur_in_shake",           "duration": 0.3,  "sfx": "whooshnormal","function": anim.blur_in_shake},
        {"animation": "rotate_3d_page_flip",     "duration": 0.5,  "sfx": "flip",        "function": anim.rotate_3d_page_flip},
        {"animation": "bounce_pop_animation",    "duration": 0.5,  "sfx": "pop1",        "function": anim.bounce_pop_animation},
        {"animation": "bubble_bounce_pop",       "duration": 0.5,  "sfx": "bpop2",       "function": anim.bubble_bounce_pop},
        {"animation": "transparent_in",          "duration": 0.5,  "sfx": "darkwoosh",   "function": anim.transparent_in},
        {"animation": "place_in_y",              "duration": 0.5,  "sfx": "swoosh",      "function": anim.place_in_y},
        {"animation": "place_in_z",              "duration": 0.5,  "sfx": "swoosh",      "function": anim.place_in_z},
    ]

    for seg in dialogues:
        img_path = seg.get("image_download_path")
        if not img_path or not os.path.exists(img_path):
            seg["mov_path"] = None
            continue

        ext = os.path.splitext(img_path)[1].lower()
        base = os.path.splitext(os.path.basename(img_path))[0]

        # Skip animation generation for GIFs; just assign the path directly
        if ext == ".gif":
            seg["mov_path"] = img_path
            continue

        # Look for existing .mov to reuse
        existing_mov = next(
            (f for f in os.listdir(MOV_DIR) if f.startswith(base + "_") and f.endswith(".mov")),
            None
        )
        if existing_mov:
            seg["mov_path"] = os.path.join(MOV_DIR, existing_mov)
            continue

        # Case: Image → Animate and render
        fx = random.choice(fx_defs)
        mov_name = f"{base}_{fx['animation']}.mov"
        output_mov_path = os.path.join(MOV_DIR, mov_name)
        sfx_file = os.path.join(SFX_DIR, fx["sfx"] + ".mp3") if fx.get("sfx") else None

        fx["function"](img_path, duration=fx["duration"], sfx=sfx_file)
        anim.render_mov(mov_name)

        seg["mov_path"] = output_mov_path

    state["audio_dialogues"] = dialogues
    return {
        "audio_dialogues": dialogues,
    }



async def pull_from_drive(state: ScriptState) -> ScriptState:
    folder_name = state['topic']
    audio_path_base = os.path.join(files_dir, folder_name, 'audios')
    os.makedirs(audio_path_base, exist_ok=True)

    service = authenticate_google_drive()

    root_id = 'root'
    ca_id = find_folder_id_by_name(service, 'channel_automations', root_id)
    pa_id = find_folder_id_by_name(service, 'peterAI', ca_id)
    scripts_id = find_folder_id_by_name(service, 'SCRIPTS', pa_id)
    topic_id = find_folder_id_by_name(service, folder_name, scripts_id)

    if not all([ca_id, pa_id, scripts_id, topic_id]):
        raise RuntimeError("❌ Required folder structure not found.")

    for speaker in ['stewie', 'peter']:
        speaker_drive_id = find_folder_id_by_name(service, speaker, topic_id)
        if not speaker_drive_id:
            print(f"⚠️ Folder '{speaker}' not found in Drive topic folder. Skipping.")
            continue

        speaker_local_path = os.path.join(audio_path_base, speaker)
        os.makedirs(speaker_local_path, exist_ok=True)

        print(f"\n📁 Downloading files for: {speaker}")
        files = list_files(service, speaker_drive_id)
        if not files:
            print(f"❌ No files found in speaker folder: {speaker}")
            continue

        for f in files:
            file_name = f["name"]
            local_path = os.path.join(speaker_local_path, file_name)

            if os.path.exists(local_path):
                print(f"✅ Skipping (already exists): {file_name}")
                continue

            print(f"⬇️  Downloading: {file_name} → {local_path}")
            download_file(service, f["id"], local_path)

    print("\n✅ All speaker folders downloaded.")
    return {'audio_path_base': audio_path_base}




def get_audio_duration(file_path: str) -> float:
    """Return duration of an audio file in seconds using ffmpeg.probe."""
    probe = ffmpeg.probe(file_path)
    return float(probe['format']['duration'])

def extract_line_number(filename: str) -> int:
    """Extracts the line number from a filename like 'peter_line4.wav'."""
    match = re.search(r'_line(\d+)\.wav$', filename)
    return int(match.group(1)) if match else -1

async def process_audio_create_structure(state: ScriptState) -> ScriptState:
    folder_name = state['topic']
    audio_path_base = state['audio_path_base']
    audios_path = os.path.join(audio_path_base)
    os.makedirs(audios_path, exist_ok=True)
    script_output_dir_path = state['script_output_dir_path']

    dialogue = state.get('audio_dialogues', [])  
    combined_dialogues: List[DialogueLineAudio] = []
    current_time = 0.0
    audio_files_in_order: List[str] = []

    speaker_file_map = defaultdict(dict)
    for speaker_folder in os.listdir(audios_path):
        speaker_path = os.path.join(audios_path, speaker_folder)
        if not os.path.isdir(speaker_path):
            continue
        for filename in os.listdir(speaker_path):
            if filename.endswith('.wav'):
                line_num = extract_line_number(filename)
                if line_num == -1:
                    continue
                speaker_file_map[speaker_folder.lower()][line_num] = os.path.join(speaker_path, filename)

    for i, line in enumerate(dialogue):
        speaker = line['speaker'].lower()
        expected_line_num = i + 1
        file_path = speaker_file_map.get(speaker, {}).get(expected_line_num)

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"No matching audio file for {speaker}_line{expected_line_num}.wav")

        duration = get_audio_duration(file_path)
        start = current_time
        end = current_time + duration
        current_time = end
        audio_files_in_order.append(file_path)

        combined_dialogues.append({
            'speaker': line['speaker'],
            'text': line['text'],
            'search_query': line.get('search_query'),
            'speaker_img': line.get('speaker_img'),
            'image_uri': line.get('image_uri'),
            'image_download_path': line.get('image_download_path'),
            'mov_path': line.get('mov_path'),  # ✅ include existing mov_path
            'audio_name': os.path.basename(file_path),
            'file_path': file_path,
            'start': start,
            'end': end,
            'duration': duration,
        })

    concat_list_path = os.path.join(audios_path, f'{folder_name}_concat_list.txt')
    combined_audio_path = os.path.join(audios_path, f'{folder_name}.wav')

    with open(concat_list_path, 'w') as f:
        for path in audio_files_in_order:
            f.write(f"file '{os.path.abspath(path)}'\n")

    ffmpeg.input(concat_list_path, format='concat', safe=0).output(
        combined_audio_path, acodec='pcm_s16le', ac=1, ar='44100'
    ).overwrite_output().run()

    print(f"Combined audio saved to: {combined_audio_path}")

    state['audio_dialogues'] = combined_dialogues
    state['combined_audio_path'] = combined_audio_path
    state['complete_duration'] = current_time

    file_path = os.path.join(script_output_dir_path, "script.json")
    os.makedirs(script_output_dir_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"topic": folder_name, "audio_dialogues": combined_dialogues}, f, indent=2, ensure_ascii=False)

    return {
        'audio_dialogues': combined_dialogues,
        'combined_audio_path': combined_audio_path,
        'complete_duration': current_time
    }

def mutate_gif_dims(gif_file, h, w, padding):
    """
    Resize a GIF to fit within (w - 2*padding, h - 2*padding),
    and pad it to (w x h) with transparent background.
    Returns the mutated .gif file path.
    """
    base_name = os.path.splitext(gif_file)[0]
    output_file = f"{base_name}_mutated_{w}x{h}.gif"

    if os.path.exists(output_file):
        return output_file

    target_width = w - int(2 * padding)
    target_height = h - int(2 * padding)

    # Input stream
    input_stream = ffmpeg.input(gif_file)

    # Scale + pad
    scaled = (
        input_stream
        .filter('scale', target_width, target_height, force_original_aspect_ratio='decrease')
        .filter('pad', w, h, '(ow-iw)/2', '(oh-ih)/2', color='0x00000000')
    )

    # Generate palette for better GIF transparency handling
    palette = (
        scaled
        .filter('palettegen', stats_mode=0)
    )

    # Apply palette
    output = (
        scaled
        .filter('paletteuse', alpha_threshold=128)
    )

    # Because ffmpeg-python needs a special way to handle multiple inputs for palette use:
    out = ffmpeg.output(
        output,
        output_file,
        **{'loop': 0}
    )

    # Run the ffmpeg command with palette generation via complex filter:
    # So we need to run a single command with both scaled and palette as inputs

    # Instead, we do it via ffmpeg command line using complex filter:
    # ffmpeg -i input.gif -filter_complex "[0:v]scale=...pad=...,palettegen" palette.png
    # ffmpeg -i input.gif -i palette.png -filter_complex "[0:v]scale=...pad=...[1:v]paletteuse" output.gif

    # Since ffmpeg-python doesn't support this easily, fallback to running command line:

    import subprocess

    palette_path = f"{base_name}_palette.png"

    # Generate palette
    cmd_palette = [
        "ffmpeg", "-y", "-i", gif_file,
        "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x00000000,palettegen=stats_mode=0",
        palette_path
    ]
    subprocess.run(cmd_palette, check=True)

    # Use palette
    cmd_use_palette = [
        "ffmpeg", "-y", "-i", gif_file, "-i", palette_path,
        "-lavfi", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x00000000 [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:alpha_threshold=128",
        "-loop", "0",
        output_file
    ]
    subprocess.run(cmd_use_palette, check=True)

    # Remove palette file
    os.remove(palette_path)

    return output_file


async def overlay_anims_and_images(state: ScriptState) -> ScriptState:
    import os, random
    import ffmpeg

    SLIDE_DURATION = 0.15
    VIDEO_WIDTH = 1080
    VIDEO_HEIGHT = 1920
    CHAR_IMG_HEIGHT = 750
    CHAR_IMG_Y = VIDEO_HEIGHT - CHAR_IMG_HEIGHT - 50
    HALF_SCREEN_X = 300



    script_path = state['script_output_dir_path']

    # Load script.json and extract dialogues
    script_json_path = os.path.join(script_path, "script.json")
    with open(script_json_path, 'r') as f:
        script_data = json.load(f)
    dialogues = script_data['audio_dialogues']


    # dialogues = state['audio_dialogues']
    duration = state['complete_duration']
    topic_slug = state['topic']
    audio_file = state['combined_audio_path']


    GAMEPLAY_DIR = "src/open_deep_research/RESOURCES/GAME_PLAYS"
    CHAR_IMG_DIR = "src/open_deep_research/RESOURCES/CHAR_IMGS"

    out_dir = os.path.join(script_path, 'videos')
    os.makedirs(out_dir, exist_ok=True)

    vids = [os.path.join(GAMEPLAY_DIR, f) for f in os.listdir(GAMEPLAY_DIR)
            if f.lower().endswith(('.mp4', '.mov', '.webm'))]
    if not vids:
        raise FileNotFoundError("No gameplay videos found.")
    bg = random.choice(vids)
    if float(ffmpeg.probe(bg)['format']['duration']) < duration:
        raise ValueError("Background video too short.")

    gameplay = ffmpeg.input(bg, ss=0, t=duration)
    current_video = (
        gameplay.video
        .filter('scale', VIDEO_WIDTH, VIDEO_HEIGHT, force_original_aspect_ratio='increase')
        .filter('crop', VIDEO_WIDTH, VIDEO_HEIGHT)
    )


    # Pre-clone character images
    img_counts = {}
    for dlg in dialogues:
        img = dlg.get('speaker_img')
        char = dlg.get('speaker')
        if img and char:
            path = os.path.join(CHAR_IMG_DIR, char.lower(), img)
            img_counts[path] = img_counts.get(path, 0) + 1

    char_img_clones = {}
    for path, count in img_counts.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Character image not found: {path}")
        inp = ffmpeg.input(path)
        scaled = inp.video.filter('scale', -1, CHAR_IMG_HEIGHT)
        char_img_clones[path] = scaled.filter_multi_output('split', count)
    char_img_idx = {p: 0 for p in char_img_clones}

    # Pre-clone MOV and GIF overlays
    anim_counts = {}
    for dlg in dialogues:
        mp = dlg.get('mov_path')
        if mp and mp.lower().endswith(('.mov', '.gif')):
            path = mp
            if path.lower().endswith('.gif'):
                path = mutate_gif_dims(mp, int(VIDEO_HEIGHT * SCREEN_COVERAGE), VIDEO_WIDTH, 0.01)
            anim_counts[path] = anim_counts.get(path, 0) + 1

    anim_clones = {}
    anim_idx = {}
    for path, count in anim_counts.items():
        inp = ffmpeg.input(path)
        if path.lower().endswith('.mov'):
            stream = (
                inp.video
                .filter('format', 'yuva420p')
                .filter('scale', VIDEO_WIDTH, VIDEO_HEIGHT, force_original_aspect_ratio='increase')
                .filter('crop', VIDEO_WIDTH, VIDEO_HEIGHT)
            )
        else:  # GIF
            stream = (
                inp.video.filter('format', 'yuva420p')
                          .filter('loop', loop=-1, size=32767)
            )
        anim_clones[path] = stream.filter_multi_output('split', count)
        anim_idx[path] = 0

    for dlg in dialogues:
        mp = dlg.get('mov_path')
        img_name = dlg.get('speaker_img')
        char = dlg.get('speaker')
        start = dlg['start']
        dur = dlg['duration']
        end_anim = min(start + min(7, dur), duration)
        end_char = min(start + dur, duration)

        # Overlay animation (MOV or GIF)
        overlay_vid = None
        if mp:
            path = mp
            if mp.lower().endswith('.gif'):
                path = mutate_gif_dims(mp, int(VIDEO_HEIGHT * SCREEN_COVERAGE), VIDEO_WIDTH, 0.01)
            clones = anim_clones.get(path)
            idx = anim_idx.get(path, 0)
            if clones:
                base = clones[idx]
                anim_idx[path] += 1
                delay = 0.5
                anim_start = start + delay
                overlay_vid = base.filter('setpts', f'PTS-STARTPTS+{anim_start}/TB')

        if overlay_vid:
            current_video = ffmpeg.overlay(
                current_video, overlay_vid,
                x=0, y=0,
                enable=f'between(t,{anim_start},{end_anim})'
            )

        # Overlay character image
        if img_name and char:
            path = os.path.join(CHAR_IMG_DIR, char.lower(), img_name)
            clones = char_img_clones.get(path)
            idx = char_img_idx.get(path, 0)
            if clones:
                clip = clones[idx]
                char_img_idx[path] += 1
                slide_x = HALF_SCREEN_X
                start_x = VIDEO_WIDTH if char.lower() == 'peter' else -VIDEO_WIDTH
                xexp = (
                    f"if(between(t,{start},{start+SLIDE_DURATION}),"
                    f"{start_x} + ({slide_x}-{start_x})*(t-{start})/{SLIDE_DURATION},"
                    f"{slide_x})"
                )
                img_clip = clip.filter('setpts', f'PTS-STARTPTS+{start}/TB')
                current_video = ffmpeg.overlay(
                    current_video, img_clip,
                    x=xexp, y=CHAR_IMG_Y,
                    enable=f'between(t,{start},{end_char})'
                )

    main_audio = ffmpeg.input(audio_file).audio
    out_path = os.path.join(out_dir, f"{topic_slug}_final.mp4")
    out = ffmpeg.output(
        current_video, main_audio, out_path,
        vcodec='libx264', acodec='aac', crf=18,
        preset='fast', pix_fmt='yuv420p', t=duration
    )

    ffmpeg.run(out, overwrite_output=True)
    state['video_path'] = out_path
    return {'video_path': out_path}



async def generate_subtitles(state: ScriptState) -> ScriptState:
    script_video_path = state['video_path']
    script_audio_path = state['combined_audio_path']
    script_path = state['script_output_dir_path']

    model = WhisperModel('distil-large-v3', device='cuda', compute_type='int8')

    # Tighter constraints for font size 100
    max_chars = 14
    max_words = 4
    max_duration = 1.6
    max_wordgap = 0.5

    # Step 1: transcribe with word-level timestamps
    segments, _ = model.transcribe(script_audio_path, word_timestamps=True)
    segments = list(segments)

    # Flatten word data
    word_data = [
        {'word': word.word, 'start': word.start, 'end': word.end}
        for segment in segments
        for word in segment.words
    ]

    # Step 2: group words into clean lines
    subtitles = []
    line = []
    for idx, word_info in enumerate(word_data):
        line.append(word_info)
        duration = line[-1]['end'] - line[0]['start']
        char_count = sum(len(w['word'].strip()) for w in line)
        gap_exceeded = (
            idx < len(word_data) - 1 and 
            word_data[idx + 1]['start'] - word_info['end'] > max_wordgap
        )

        if (
            duration >= max_duration or
            char_count >= max_chars or
            len(line) >= max_words or
            gap_exceeded
        ):
            if line:
                subtitle_line = {
                    "word": " ".join(w["word"].strip() for w in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"]
                }
                subtitles.append(subtitle_line)
                line = []

    # Add any leftover
    if line:
        subtitle_line = {
            "word": " ".join(w["word"].strip() for w in line),
            "start": line[0]["start"],
            "end": line[-1]["end"]
        }
        subtitles.append(subtitle_line)

    # Step 3: Save to JSON
    os.makedirs(script_path, exist_ok=True)
    output_json = os.path.join(script_path, "subtitles.json")
    with open(output_json, 'w') as f:
        json.dump(subtitles, f, indent=4)

    return {'subtitles_path': output_json}




async def overlay_captions(state: ScriptState):
    script_path = state['script_output_dir_path']
    video_path = state['video_path']
    subtitles_json_path = state['subtitles_path']
    sanitized_title = sanitize_topic_title(state['topic'])
    output_path = os.path.join(script_path, 'videos', f"CAPTIONED_{sanitized_title}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load subtitles
    with open(subtitles_json_path, 'r') as f:
        subtitles = json.load(f)
    
    captioner = VideoCaptioner()
    # Remove await since the method is no longer async
    captioner.add_captions_to_video(video_path, subtitles, output_path)
    
    return {"captioned_video_path": output_path}


async def generate_instagram_reel_metadata(state: ScriptState, config: RunnableConfig) -> ScriptState:
    configurable = WorkflowConfiguration.from_runnable_config(config)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    
    # writer_model = init_chat_model(model='deepseek-chat', model_provider='deepseek', model_kwargs=writer_model_kwargs, temperature=0.15) 
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs, temperature=0.15) 


    prompt_template = ChatPromptTemplate.from_messages([
        ("human", """
        Topic:
        {topic}

        Here is the researched content to turn into an Instagram Reel:

        {final_report}

        Based on this, generate the Reel metadata:
        - Title: hook-based, related to the main idea
        - Description: SEO-friendly, readable, includes context and hashtags
        - Catchy Text: ultra-clickable thumbnail copy (≤8 words, high-impact, 2–3 matching emojis, no clichés)
        """)
    ])

    formatted_messages = prompt_template.format_messages(
        topic=state["topic"],
        final_report=state["final_report"]
    )

    reel_meta = await writer_model.with_structured_output(ReelMetaData).ainvoke(formatted_messages)

    return {'video_metadata': reel_meta}




async def choose_best_image(llm, images: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Format descriptions for LLM prompt
    descriptions = "\n".join(f"{i}: {img['description'] or 'No description'}" for i, img in enumerate(images))
    prompt = (
        "You are an expert at choosing the best thumbnail image based on descriptions.\n"
        "Given the following image descriptions, select the best one by number and explain briefly why:\n"
        f"{descriptions}\n"
        "Reply with only the index number."
    )
    response = await llm.with_structured_output(BestImageIndex).ainvoke(prompt)
    try:
        choice_idx = int(response['index'])
        return images[choice_idx]
    except (ValueError, IndexError):
        # fallback: pick first image if parsing fails
        return images[0]


def add_text_effects(base_img, text_img, x, y, border=2, shadow_offset=(4, 4)):
    alpha = text_img.split()[-1]

    # Shadow layer
    shadow = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 160), (x + shadow_offset[0], y + shadow_offset[1]), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(2))

    # Border layer
    border_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    for dx in range(-border, border + 1):
        for dy in range(-border, border + 1):
            if dx == 0 and dy == 0:
                continue
            border_layer.paste((0, 0, 0, 255), (x + dx, y + dy), alpha)

    # Combine
    base_img = Image.alpha_composite(base_img, shadow)
    base_img = Image.alpha_composite(base_img, border_layer)
    base_img.paste(text_img, (x, y), text_img)
    return base_img


async def download_image(url):
    print(f"Downloading image from: {url}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as response:
                content_type = response.headers.get('Content-Type', '')
                print(f"Content-Type: {content_type}")
                if not content_type.startswith('image/'):
                    raise ValueError(f"URL did not return an image. Content-Type: {content_type}")
                data = await response.read()
                return Image.open(BytesIO(data)).convert("RGBA")
        except Exception as e:
            print(f"Failed to download or identify image from {url}: {e}")
            raise


async def generate_thumbnail(state: ScriptState, config: RunnableConfig) -> ScriptState:
    configurable = WorkflowConfiguration.from_runnable_config(config)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})

    # writer_model = init_chat_model(model='deepseek-chat', model_provider='deepseek', model_kwargs=writer_model_kwargs, temperature=0.15) 
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs, temperature=0.15) 


    title = state["video_metadata"]["title"]
    description = state["video_metadata"]["description"]
    catchy_text = state["video_metadata"]["catchy_text"]
    sanitized = title.replace(" ", "_")

    thumbs_dir = os.path.join(state["script_output_dir_path"], "thumbnails")
    os.makedirs(thumbs_dir, exist_ok=True)
    output_path = os.path.join(thumbs_dir, f"THUMBNAIL_{sanitized}.jpg")

    ### 1. Generate Search Query via ChatPrompt-style Instruction
    query_prompt = f"""
        You are a creative assistant helping to generate a high-impact search query for finding the perfect thumbnail image for a technical video.

        Task:
        - Generate a concise, keyword-rich search query to find a thumbnail image that visually represents the video topic.
        - Use terms like 'diagram', 'icon', 'labeled illustration'.
        - Avoid vague terms or stock photography cues.


        Title: {title}

        Strictly return the search query only.
    """

    class RetryThumbnailSearchQuery(TypedDict):
        query: str

    search_query = (await writer_model.with_structured_output(RetryThumbnailSearchQuery).ainvoke(query_prompt))['query']

    async def get_best_image(query, retries=2):
        for attempt in range(retries + 1):
            images = await tavily_image_search(query)
            if images:
                return images
            if attempt < retries:
                # Rewriting search query using writer model
                feedback = f"The query '{query}' returned no useful images. Suggest a better one."
                query = (await writer_model.with_structured_output(RetryThumbnailSearchQuery).ainvoke(feedback))['query']
        return None

    images = await get_best_image(search_query)

    if not images:
        raise RuntimeError("No images found for thumbnail generation.")
    best = await choose_best_image(writer_model, images)
    # best= images[4]

    overlay = await download_image(best['url'])
    overlay = add_white_background(overlay, padding=0)

    # Extract video frame at 0.5s
    video_path = state['video_path']
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(0.3)
    clip.close()

    frame_img = Image.fromarray(frame).convert("RGBA")
    enhancer = ImageEnhance.Brightness(frame_img)
    frame_img = enhancer.enhance(0.95)

    canvas = frame_img
    thumb_w, thumb_h = canvas.size
    top_h = int(thumb_h * 0.5)

    overlay.thumbnail((thumb_w - 20, top_h - 20), Image.Resampling.LANCZOS)
    ox = (thumb_w - overlay.width) // 2
    oy = (top_h - overlay.height) // 2 - 140
    canvas.paste(overlay, (ox, oy), overlay)

    # Font setup
    font_path = get_config_value(configurable.font_style_path)
    font = ImageFont.truetype(font_path, 130)

    text_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        words = text.split(' ')
        lines, current = [], ""
        for w in words:
            test = current + (' ' if current else '') + w
            if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
                current = test
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines

    lines = wrap_text(catchy_text, font, thumb_w - 40)
    line_height = font.getbbox("Ay")[3] + 30
    total_h = line_height * len(lines)

    # Vertically center with 5% downward shift
    y_start = int((thumb_h - total_h) // 2 + thumb_h * 0.05)

    # Caption color and border
    CAPTIONS_TEXT_COLOR = (153, 255, 204, 255)
    CAPTIONS_BORDER_COLOR = (0, 0, 0, 255)
    BORDER_THICKNESS = 10

    emoji_source = GoogleEmojiSource()
    with Pilmoji(text_layer, source=emoji_source) as pilmoji:
        for i, line in enumerate(lines):
            y = y_start + i * line_height
            x = (thumb_w - draw.textbbox((0, 0), line, font=font)[2]) // 2

            # Border (thick)
            for dx in range(-BORDER_THICKNESS, BORDER_THICKNESS + 1):
                for dy in range(-BORDER_THICKNESS, BORDER_THICKNESS + 1):
                    if dx == 0 and dy == 0:
                        continue
                    pilmoji.text((x + dx, y + dy), line, font=font, fill=CAPTIONS_BORDER_COLOR)

            pilmoji.text((x, y), line, font=font, fill=CAPTIONS_TEXT_COLOR)

    final = Image.alpha_composite(canvas, text_layer)
    final.convert("RGB").save(output_path, format="JPEG")

    return {'thumbnail_path': output_path}


async def save_final_state_json(state: ScriptState) -> ScriptState:
    """
    Saves the full ScriptState as a FINAL_STATE_<topic>.json file
    at the path defined by state['script_output_dir_path'].
    """
    topic_slug = state['topic']
    output_dir = state['script_output_dir_path']
    os.makedirs(output_dir, exist_ok=True)

    json_filename = f"FINAL_STATE_{topic_slug}.json"
    json_path = os.path.join(output_dir, json_filename)

    # Convert non-serializable items if needed (like Path, PosixPath)
    def convert(obj):
        if hasattr(obj, '__str__'):
            return str(obj)
        return obj

    with open(json_path, 'w') as f:
        json.dump(state, f, indent=2, default=convert)

    print(f"✅ Final state saved to: {json_path}")
    return {'final_state_json_path': json_path}




# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, output=ReportStateOutput)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)


# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")




# # Script state

script_builder = StateGraph(ScriptState, input=ReportStateInput, output=ScriptOutputState, config_schema=WorkflowConfiguration)
script_builder.add_node('open_deep_researcher', builder.compile())
script_builder.add_node('script_generator', script_generator)
script_builder.add_node('push_for_cloning', push_for_cloning)
script_builder.add_node('decide_images', decide_images)
script_builder.add_node('create_movs', create_movs)

script_builder.add_node('pull_from_drive', pull_from_drive)
script_builder.add_node('process_audio_create_structure', process_audio_create_structure)
script_builder.add_node('overlay_anims_and_images', overlay_anims_and_images)
script_builder.add_node("generate_subtitles", generate_subtitles)
script_builder.add_node("overlay_captions", overlay_captions)
script_builder.add_node("generate_instagram_reel_metadata", generate_instagram_reel_metadata)
script_builder.add_node("generate_thumbnail", generate_thumbnail)
script_builder.add_node("save_final_state_json", save_final_state_json)







script_builder.add_edge(START, 'open_deep_researcher')
script_builder.add_edge('open_deep_researcher', 'script_generator')
script_builder.add_edge('script_generator', 'push_for_cloning')
script_builder.add_edge('push_for_cloning', 'decide_images')
script_builder.add_edge('decide_images', 'create_movs')
script_builder.add_edge('create_movs', 'pull_from_drive')
script_builder.add_edge('pull_from_drive', 'process_audio_create_structure')
script_builder.add_edge('process_audio_create_structure', 'overlay_anims_and_images')
script_builder.add_edge('overlay_anims_and_images', 'generate_subtitles')
script_builder.add_edge('generate_subtitles', 'overlay_captions')
script_builder.add_edge('overlay_captions', 'generate_instagram_reel_metadata')
script_builder.add_edge('generate_instagram_reel_metadata', 'generate_thumbnail')
script_builder.add_edge('generate_thumbnail', 'save_final_state_json')
script_builder.add_edge('save_final_state_json', END)






graph = script_builder.compile()





