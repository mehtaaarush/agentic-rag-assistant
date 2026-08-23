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
import time
import json
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config ---
FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-flash-latest"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# Load the retrieval stack (same embeddings used at ingestion)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = FAISS.load_local(
    FAISS_PATH, embeddings, allow_dangerous_deserialization=True
)


def call_llm(prompt):
    """Single Gemini call with retry for rate limits and temporary server errors."""
    for attempt in range(5):
        try:
            resp = client.models.generate_content(model=LLM_MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                print("  Rate limit hit, waiting 30s...")
                time.sleep(30)
            elif "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower():
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s... backoff
                print(f"  Model busy (503), waiting {wait}s and retrying...")
                time.sleep(wait)
            else:
                raise
    print("  Failed after retries.")
    return ""


def get_answer_and_contexts(question, k=4):
    """Retrieve chunks and generate an answer (our RAG pipeline)."""
    docs = vector_store.similarity_search(question, k=k)
    contexts = [d.page_content for d in docs]
    context_text = "\n\n".join(contexts)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't have enough information.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""
    answer = call_llm(prompt)
    return answer, contexts


def judge_faithfulness(answer, contexts):
    """LLM-as-judge: is the answer supported by the retrieved context?"""
    context_text = "\n\n".join(contexts)
    prompt = f"""You are evaluating whether an ANSWER is faithful to the provided CONTEXT.
An answer is faithful if all its factual claims can be verified from the context.
Rate faithfulness from 0.0 (fully unsupported/hallucinated) to 1.0 (fully supported).
Respond with ONLY a number between 0.0 and 1.0.

CONTEXT:
{context_text}

ANSWER:
{answer}

FAITHFULNESS SCORE (0.0-1.0):"""
    score = call_llm(prompt)
    try:
        return float(score.split()[0])
    except (ValueError, IndexError):
        return None


def judge_relevancy(question, answer):
    """LLM-as-judge: does the answer actually address the question?"""
    prompt = f"""You are evaluating whether an ANSWER is relevant to a QUESTION.
Rate relevancy from 0.0 (completely off-topic) to 1.0 (directly and fully answers the question).
Respond with ONLY a number between 0.0 and 1.0.

QUESTION:
{question}

ANSWER:
{answer}

RELEVANCY SCORE (0.0-1.0):"""
    score = call_llm(prompt)
    try:
        return float(score.split()[0])
    except (ValueError, IndexError):
        return None


# --- Evaluation question set ---
eval_questions = [
    "What is the shifted window mechanism in the Swin Transformer?",
    "How does DINOv2 learn visual features?",
    "What is the core idea of a Generative Adversarial Network?",
    "What problem does the self-attention mechanism solve in transformers?",
    "What is CLIP and how is it trained?",
]


def main():
    results = []
    print("Running custom LLM-as-judge evaluation...\n")

    for i, question in enumerate(eval_questions, 1):
        print(f"[{i}/{len(eval_questions)}] {question}")
        answer, contexts = get_answer_and_contexts(question)
        time.sleep(2)  # gentle pacing to respect rate limits

        faith = judge_faithfulness(answer, contexts)
        time.sleep(2)
        rel = judge_relevancy(question, answer)
        time.sleep(2)

        print(f"    Faithfulness: {faith}   Relevancy: {rel}\n")
        results.append({
            "question": question,
            "answer": answer,
            "faithfulness": faith,
            "relevancy": rel,
        })

    # Averages (ignoring any failed scores)
    faith_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    rel_scores = [r["relevancy"] for r in results if r["relevancy"] is not None]

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0

    print("=" * 50)
    print("EVALUATION RESULTS (LLM-as-judge)")
    print("=" * 50)
    print(f"Average Faithfulness: {avg_faith:.2f}")
    print(f"Average Answer Relevancy: {avg_rel:.2f}")
    print("=" * 50)

    # Save for your README / screenshots
    with open("eval_results.json", "w") as f:
        json.dump({
            "average_faithfulness": round(avg_faith, 3),
            "average_relevancy": round(avg_rel, 3),
            "per_question": results,
        }, f, indent=2)
    print("\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    main()