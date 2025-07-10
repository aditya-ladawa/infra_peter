# from qdrant_client import QdrantClient
# from langchain.chat_models import init_chat_model
# from qdrant_client.http import models
# from qdrant_client.http.models import UpdateStatus
# from qdrant_client.models import Distance, VectorParams
# import re
# from langchain.schema import Document
# from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
# from qdrant_client import QdrantClient, models
# from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

# from uuid import uuid4

# from langchain_core.documents import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time
# from langchain_core.prompts import ChatPromptTemplate
# from uuid import uuid4
# load_dotenv()


# # EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-exp-03-07')
# EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

# sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# llm_for_retrievel = init_chat_model(model='gemini-2.0-flash-lite-001', model_provider='google_vertexai', temperature=0)

# COLLECTION_NAME = 'docs'
# VECTOR_SIZE = 768



# def connect_and_create_collection(COLL_NAME: str, SIZE: int):

#     try:
#         client = QdrantClient(url="http://localhost:6333")
#         print('\nStarted Qdrant client.')


#         existing_collections = client.get_collections().collections
#         if COLL_NAME not in [collection.name for collection in existing_collections]:
#             client.create_collection(
#                 collection_name=COLL_NAME,
#                 vectors_config={"dense": VectorParams(size=SIZE, distance=Distance.COSINE)},
#                 sparse_vectors_config={
#                     "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
#                 },
#             )

#             client.create_payload_index(
#                 collection_name=COLL_NAME,
#                 field_name="metadata.section",
#                 field_schema=models.PayloadSchemaType.KEYWORD,
#             )

#             print("\nIndex created on 'section' field for efficient filtering.\n")
#             print(f"Collection '{COLL_NAME}' created successfully.\n")
#         else:
#             print(f"Collection '{COLL_NAME}' already exists.")

#     except Exception as e:
#         print(f"Connection error: {e}")
#         client = None
#     return client


# def manual_markdown_chunker_with_headers_skip_empty(md_text: str, file_path: str) -> list[Document]:
#     header_pattern = re.compile(r"^(#{1,3})\s+(.*)", re.MULTILINE)
#     matches = list(header_pattern.finditer(md_text))

#     documents = []
#     hierarchy = {"section": None, "subsection": None, "topic": None}

#     for i, match in enumerate(matches):
#         level = len(match.group(1))
#         title = match.group(2).strip()

#         start = match.start()
#         end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)

#         chunk = md_text[start:end].strip()

#         # Skip chunk if only header is present, no other content after it (excluding possible links)
#         # We'll remove the header line and check if anything else remains.
#         chunk_lines = chunk.splitlines()
#         if len(chunk_lines) <= 1:
#             # Only header line present, skip
#             continue
#         # Also skip if after removing header line the rest is empty or just whitespace
#         body = "\n".join(chunk_lines[1:]).strip()
#         if not body:
#             continue

#         # Update hierarchy
#         if level == 1:
#             hierarchy["section"] = title
#             hierarchy["subsection"] = None
#             hierarchy["topic"] = None
#         elif level == 2:
#             hierarchy["subsection"] = title
#             hierarchy["topic"] = None
#         elif level == 3:
#             hierarchy["topic"] = title

#         metadata = {
#             "source": file_path,
#             **{k: v for k, v in hierarchy.items() if v is not None}
#         }

#         documents.append(Document(page_content=chunk, metadata=metadata))

#     return documents




# def situate_context(CHUNK: str):
#     CHUNK_CONTEXT_PROMPT = """
#     You will be given a chunk from the Manim Community Python library documentation.

#     Your task:

#     1. Provide a concise but complete context explaining where this chunk fits in the overall documentation. Specify the relevant module, class, function, or concept, and its relation to other parts.

#     2. Summarize all important information, including key details like parameters, usage, and purpose. Keep it clear and compact but do not omit critical technical content.

#     Write in a concise manner while retaining all essential details.

#     OUTPUT FORMAT:
#     Context: <your concise contextual explanation>
#     Summary: <your concise yet detailed summary>

#     Respond only in this format without extra commentary or formatting.
#     """





#     HUMAN_PROMPT = """
#     Here is the chunk we want to situate within the whole document:
#     <chunk>
#     {chunk_content}
#     </chunk>

#     Here is the documentation:
#     <documentation>
#     {doc_content}
#     </documentation>
#     """

#     CHUNK_CONTEXT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
#         ('system', CHUNK_CONTEXT_PROMPT),
#         ('human', HUMAN_PROMPT)
#     ])

#     path = "src/open_deep_research/mcdc.md"
#     with open(path, "r", encoding="utf-8") as f:
#         markdown = f.read()

#     formatted_messages = CHUNK_CONTEXT_PROMPT_TEMPLATE.format_messages(
#         chunk_content=CHUNK,
#         doc_content=markdown
#     )
#     contextual_chunk = llm_for_retrievel.invoke(formatted_messages)

#     contextual_chunk_content = contextual_chunk.content
#     return contextual_chunk_content




# def embed_material():
#     client = connect_and_create_collection(COLL_NAME=COLLECTION_NAME, SIZE=VECTOR_SIZE)

#     qdrant = QdrantVectorStore(
#         client=client,
#         collection_name=COLLECTION_NAME,
#         embedding=EMBEDDING_MODEL,
#         sparse_embedding=sparse_embeddings,
#         retrieval_mode=RetrievalMode.HYBRID,
#         vector_name="dense",
#         sparse_vector_name="sparse",
#     )

#     path = "src/open_deep_research/mcdc.md"
#     with open(path, "r", encoding="utf-8") as f:
#         markdown = f.read()

#     raw_documents = manual_markdown_chunker_with_headers_skip_empty(markdown, path)
#     enriched_documents = []

#     print(f"[INFO] Total chunks to process: {len(raw_documents)}")

#     for idx, doc in enumerate(raw_documents):
#         chunk_text = doc.page_content
#         situated_context_chunk = situate_context(chunk_text)  # LLM-generated

#         context_lines = situated_context_chunk.strip().split("\n", 1)
#         context = context_lines[0].strip() if len(context_lines) > 0 else ""
#         summary = context_lines[1].strip() if len(context_lines) > 1 else ""

#         full_text_to_embed = f"{context}\n\n{summary}\n\n{chunk_text}"

#         enriched_metadata = {
#             **doc.metadata,
#             "raw_chunk": chunk_text,
#             "context": context,
#             "summary": summary
#         }

#         enriched_documents.append(
#             Document(page_content=full_text_to_embed, metadata=enriched_metadata)
#         )

#         print(f"[INFO] Processed chunk {idx + 1}/{len(raw_documents)}")

#     uuids = [str(uuid4()) for _ in range(len(enriched_documents))]
#     qdrant.add_documents(documents=enriched_documents, ids=uuids)

#     print(f"[SUCCESS] Embedded {len(enriched_documents)} chunks into Qdrant collection '{COLLECTION_NAME}'.")


# # # # === Example usage ===
# # if __name__ == "__main__":
# #   embed_material()

# client = connect_and_create_collection(COLL_NAME=COLLECTION_NAME, SIZE=VECTOR_SIZE)

# qdrant = QdrantVectorStore(
#     client=client,
#     collection_name=COLLECTION_NAME,
#     embedding=EMBEDDING_MODEL,
#     sparse_embedding=sparse_embeddings,
#     retrieval_mode=RetrievalMode.HYBRID,
#     vector_name="dense",
#     sparse_vector_name="sparse",
# )


# retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 3})

