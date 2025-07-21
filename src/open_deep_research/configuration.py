import os
from enum import Enum
from dataclasses import dataclass, fields, field
from typing import Any, Optional, Dict, Literal

from langchain_core.runnables import RunnableConfig

from open_deep_research.prompts import SCRIPT_GEN_PROMPT, IMAGE_DECIDE_PROMPT


DEFAULT_REPORT_STRUCTURE = """
Structure your report strictly as follows with a 350-word limit:

Purpose:
One sentence defining the topic.
One sentence on stakes/impact.

1. Introduction

What: 1–2 sentences, precise definition.

Why it matters: 1–2 sentences on stakes or impact.

2. Technical Deep Dive
For 1–2 key components/issues only (150–250 words total):

[Component/Issue Title]
Summary: 1 sentence core insight.

Body:

Problem: Clear bottleneck or failure.

Solutions evaluated: 3–5 bullet concise list.

Chosen solution: Selection rationale, pros/cons.

Tradeoffs: Bullets or table with costs/risks.

Outcome: Concrete metric or impact.

Viral Analogy: Short, sharp metaphor capturing the problem/tradeoff.
Controversy/Myth to Bust: Common misconception relevant here.

Key Insight: One lesson with systemic value.

3. Patterns & Principles
Markdown table or bullet list, max 5 entries:

Pattern	Context	When to Use	Tradeoffs
Example Pattern 1	Relevant context	Conditions for use	Key costs or risks
...	...	...	...

4. Summary

Narrative recap: ≤100 words, factual, no opinion.

Key takeaways: 3–5 crisp bullets.

Sarcasm/Contrarian angle: Brief note for script tone.

Final line: Sharp systemic conclusion.

Notes:

Use minimal jargon; prefer clarity over fluff.

Prioritize one or two sharp technical insights only.

Avoid unnecessary verbosity; word economy is mandatory.

This format targets script LLM input, not human readability or engagement.

Limit total length to 350 words by cutting all non-critical elaboration.
"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"
    NONE = "none"

@dataclass(kw_only=True)
class WorkflowConfiguration:
    """Configuration for the workflow/graph-based implementation (graph.py)."""
    # # my extensions
    script_gen_system_prompt: str = SCRIPT_GEN_PROMPT
    image_decide_prompt:str = IMAGE_DECIDE_PROMPT

    font_style_path: str = 'Protest_Strike/ProtestStrike-Regular.ttf'
    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "google_genai"
    summarization_model: str = "gemini-2.5-flash"
    max_structured_output_retries: int = 3
    include_source_str: bool = False
    
    # Workflow-specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: str = "google_genai"
    planner_model: str = 'gemini-2.5-flash'
    planner_model_kwargs: Optional[Dict[str, Any]] = None
    writer_provider: str = "google_genai"
    writer_model: str = "gemini-2.5-flash"
    writer_model_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

@dataclass(kw_only=True)
class MultiAgentConfiguration:
    """Configuration for the multi-agent implementation (multi_agent.py)."""
    # Common configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "google_genai"
    summarization_model: str = 'gemini-2.5-flash'
    include_source_str: bool = False
    
    # Multi-agent specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per section
    supervisor_model: str = 'gemini-2.5-flash'
    researcher_model: str = 'gemini-2.5-flash'
    ask_for_clarification: bool = False # Whether to ask for clarification from the user
    # MCP server configuration
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[list[str]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MultiAgentConfiguration":
        """Create a MultiAgentConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# Keep the old Configuration class for backward compatibility
Configuration = WorkflowConfiguration
