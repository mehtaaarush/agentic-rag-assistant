import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config ---
DATA_PATH = "data"                    # where the PDFs live
FAISS_PATH = "faiss_index"            # where we'll save the vector store
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Load every PDF in the data folder and extract its text."""
    print("Loading PDFs from the data folder...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs.")
    return documents


def split_documents(documents):
    """Break the loaded pages into smaller overlapping chunks."""
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # each chunk is ~1000 characters
        chunk_overlap=200,     # consecutive chunks share 200 characters
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks):
    """Embed each chunk and store all vectors in a FAISS index on disk."""
    print("Loading the embedding model (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and building the FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(FAISS_PATH)
    print(f"Vector store saved to '{FAISS_PATH}'.")


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    build_vector_store(chunks)
    print("\nIngestion complete. Your knowledge base is ready.")


if __name__ == "__main__":
    main()