import warnings
import os
import logging

# Silence cosmetic warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger().setLevel(logging.ERROR)
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# --- Config ---
FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-flash-latest"

# --- Load API keys ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in .env")

# --- Load the vector store once, at startup ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)


# --- TOOL 1: Search the papers (your RAG retrieval) ---
@tool
def search_papers(query: str) -> str:
    """Search the local machine-learning research papers for information.
    Use this for questions about model architectures, methods, and concepts
    covered in the ingested papers (transformers, Swin, DINOv2, GANs, CLIP, etc.)."""
    results = vector_store.similarity_search(query, k=4)
    if not results:
        return "No relevant information found in the papers."
    # Format each chunk with its source so the agent can cite it
    formatted = "\n\n---\n\n".join(
        f"[Source: {chunk.metadata.get('source', 'unknown')}, "
        f"page {chunk.metadata.get('page', '?')}]\n{chunk.page_content}"
        for chunk in results
    )
    return formatted


# --- TOOL 2: Search the web (for anything the papers don't cover) ---
web_search = TavilySearch(max_results=3)


@tool
def search_web(query: str) -> str:
    """Search the live web for current information.
    Use this for recent developments, news, or anything NOT covered in the
    local research papers (e.g. models or events after the papers were written)."""
    results = web_search.invoke({"query": query})
    return str(results)


# --- Build the agent ---
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

tools = [search_papers, search_web]

SYSTEM_PROMPT = """You are a research assistant specializing in computer vision and deep learning.

You have two tools:
1. search_papers — searches a local library of ML research papers. Prefer this for questions about model architectures, methods, and concepts.
2. search_web — searches the live web. Use this only when the papers likely don't cover the topic (recent news, models released after the papers, current events).

Always decide which tool fits the question best. After getting results, answer clearly and cite your sources (paper filenames or web URLs). If neither tool gives a useful answer, say so honestly."""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


def ask_agent(question):
    """Send a question to the agent and print the final answer + the tools it used."""
    print("\nAgent is thinking (deciding which tool to use)...\n")

    # The agent streams its steps; we collect them
    result = agent.invoke({"messages": [("user", question)]})

    # The full message history shows the agent's reasoning + tool calls
    messages = result["messages"]

    # Print which tools were called (so you can SEE the agent deciding)
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"  -> Agent chose tool: {tc['name']}  (query: {tc['args'].get('query', '')})")

    # The last message is the agent's final answer
    final_answer = messages[-1].content
    if isinstance(final_answer, list):
        final_answer = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in final_answer
        )
    print("\n" + final_answer)


def main():
    print("Agentic RAG Assistant ready. Ask a question (or type 'quit').\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        ask_agent(question)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()