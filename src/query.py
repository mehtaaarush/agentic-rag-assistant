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
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config ---
FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-flash-latest"

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

# Create the Gemini client
client = genai.Client(api_key=api_key)


def load_vector_store():
    """Load the FAISS index we built during ingestion."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created this file ourselves
    )
    return vector_store


def retrieve_context(vector_store, question, k=4):
    """Find the k most relevant chunks for the question."""
    results = vector_store.similarity_search(question, k=k)
    return results


def build_prompt(question, context_chunks):
    """Assemble the retrieved chunks and question into a grounded prompt."""
    # Combine the chunk texts, labeling each with its source
    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk.metadata.get('source', 'unknown')}, "
        f"page {chunk.metadata.get('page', '?')}]\n{chunk.page_content}"
        for chunk in context_chunks
    )

    prompt = f"""You are a helpful research assistant answering questions about machine learning papers.
Use ONLY the context below to answer the question. If the answer is not in the context, say "I don't have enough information in the provided documents to answer that."
Always cite the source filename(s) you used at the end of your answer.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""
    return prompt


def answer_question(question):
    """Full RAG flow: retrieve, build prompt, generate answer."""
    vector_store = load_vector_store()
    context_chunks = retrieve_context(vector_store, question)
    prompt = build_prompt(question, context_chunks)

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text


def main():
    print("RAG Assistant ready. Ask a question about your papers (or type 'quit').\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        print("\nThinking...\n")
        answer = answer_question(question)
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()