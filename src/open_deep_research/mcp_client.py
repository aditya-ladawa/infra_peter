from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model='deepseek-chat', model_provider='deepseek', temperature=0.0) 

client = MultiServerMCPClient(
    {
        "RAG": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["src/open_deep_research/mcp_server.py"],
            "transport": "stdio",
        },
    }
)
import asyncio

async def main():
    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    graph = builder.compile()

    r = graph.invoke({"messages": "Provide code to create animation of bouncing ball using manim"})
    print(r['messages'][-1].content)

asyncio.run(main())
