from open_deep_research.vector_store_creation import retriever
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("RAG")

@mcp.tool()
def retrieve(query: str) -> list:
    """Uses Qdrant retriever to retrieve docs from vector database"""
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs


if __name__ == "__main__":
    mcp.run(transport="stdio")