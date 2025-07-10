import os
from enum import Enum
from dataclasses import dataclass, fields, field
from typing import Any, Optional, Dict, Literal

from langchain_core.runnables import RunnableConfig

from open_deep_research.prompts import SCRIPT_GEN_PROMPT, IMAGE_DECIDE_PROMPT


DEFAULT_REPORT_STRUCTURE = """
Purpose: Pure technical documentation for downstream content creation (scripts, reels, etc.). No fluff, no alignment with channel tone—just research-to-outline fidelity.

Note:
Do not wrapping words inside special charachters - astericks ('*')

1. Introduction (Strictly Factual)
What: 1–2 sentences defining the topic (e.g., "WebSockets enable real-time client-server communication.")

Why It Matters: 1–2 sentences on stakes (e.g., "Critical for chat apps, but scaling them breaks naive implementations.")

No opinions, no narrative. Example:

"This report examines WebSocket architectures. At scale, connection handling and state synchronization become bottlenecks."

2. Technical Deep Dive (Modular Subsections)
Format per subsection:

Title: ### [Specific Component/Problem] (e.g., ### Connection Pooling)

Summary: 1-sentence TL;DR (e.g., "Optimizes server resource usage for persistent connections.")

Body (150–250 words):

Problem: What broke or needed solving? (e.g., "10K concurrent WebSockets crashed Node.js servers.")

Solutions Evaluated: Bullet list of options (e.g., - HAProxy - NGINX - Custom ELB)

Chosen Solution: Why? (e.g., "NGINX won for its built-in WebSocket support and low config overhead.")

Tradeoffs: Table or bullets (e.g., - Latency (+15ms) - Cost: ($0.02/GB data)

Outcome: Metric or result (e.g., "Handled 50K connections with 0.1% error rate.")

Rule: Each subsection must end with:

Key Insight: [1 generalizable lesson] (e.g., "Stateful protocols demand proxy-aware design.")

3. Patterns & Principles (Optional but Preferred)
Markdown table or bullets:

Pattern	Context	When to Use	Tradeoffs
Backpressure	High-volume event streams	Prevent client crashes	Adds latency
Circuit Breaker	Microservice dependencies	Avoid cascading fails	False positives
4. Summary (Robot-Readable)
Narrative Recap: 100 words max. Example:

*"WebSocket scaling requires proxy-tier connection management. NGINX reduced errors by 92%, but introduced 15ms latency. Backpressure is non-negotiable for bursty traffic."*

Key Takeaways: 3–5 bullet points (e.g., - Use proxies for connection pooling - Monitor for silent fails).

Final Line: "Long-term, this pattern shifts state management to infrastructure."


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

    font_style_path: str = 'Bubblegum_Sans/BubblegumSans-Regular.ttf'
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
    max_search_depth: int = 1 # Maximum number of reflection + search iterations
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
