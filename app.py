import os
from langchain.chat_models import init_chat_model

import dotenv
dotenv.load_dotenv()


model = init_chat_model("google_genai:gemini-2.5-flash-lite")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

from langchain_community.document_loaders import PyMuPDFLoader

# file_path = "UniversityPhysicsVolume1-LR.pdf"
# loader = PyMuPDFLoader(file_path)

# docs = loader.load()

# print(docs[0])


# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,  # chunk size (characters)
#     chunk_overlap=200,  # chunk overlap (characters)
#     add_start_index=True,  # track index in original document
# )
# all_splits = text_splitter.split_documents(docs)

# print(f"Split blog post into {len(all_splits)} sub-documents.")

# document_ids = vector_store.add_documents(documents=all_splits)

# print(document_ids[:3])

from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message
from langchain.agents import AgentState, create_agent

agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "what can reveal whether flow is laminar turbulent"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()