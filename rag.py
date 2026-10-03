import os
import json
from pathlib import Path

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "datas"
DOCS_PATH = DATA_DIR / "docs.json"
CHROMA_PATH = DATA_DIR / "chroma"

COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-5-mini"
TOP_K = 3
EMBED_BATCH_SIZE = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

openai_client = OpenAI(
    api_key=os.getenv("AI_TUNNEL_API_KEY"),
    base_url="https://api.aitunnel.ru/v1/",
)


def load_documents(path: Path = DOCS_PATH) -> list[str]:
    """Load a list of documents from a JSON file in list[str] format."""
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
        raise ValueError(f"expected list[str] in {path}")
    return docs


def chunk_documents(
    documents: list[str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split documents into chunks with the given size and overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[str] = []
    for doc in documents:
        if not doc or not doc.strip():
            continue
        chunks.extend(splitter.split_text(doc))
    return chunks


def get_collection():
    """Return (or create) the Chroma collection under DATA_DIR/chroma."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Build embeddings for a list of texts via the OpenAI API."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def index_corpus(collection, documents: list[str]) -> None:
    """Chunk documents, embed them, and index into the Chroma collection."""
    if collection.count() > 0:
        print(f"collection already indexed: {collection.count()} chunks")
        return

    chunks = chunk_documents(documents)
    print(f"indexing {len(chunks)} chunks from {len(documents)} documents into Chroma...")
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        embeddings = embed_texts(batch)
        ids = [str(start + i) for i in range(len(batch))]
        collection.add(ids=ids, documents=batch, embeddings=embeddings)
        print(f"  {min(start + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")


def retrieve(collection, question: str, k: int = TOP_K) -> list[str]:
    """Retrieve the top-k most relevant chunks for the question embedding."""
    query_embedding = embed_texts([question])[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=k)
    return result["documents"][0]


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate an LLM answer grounded strictly in the given contexts."""
    context_block = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, 1))
    llm_kwargs = {}

    response = openai_client.responses.create(
        model=GEN_MODEL,
        instructions=(
            "Answer the question using only the provided context. "
            "If the answer is not in the context, say so. Prefer a brief answer. "
            "Use the context to answer, but do not mention the context in the reply."
        ),
        input=f"Context:\n{context_block}\n\nQuestion: {question}",
        **llm_kwargs,
    )
    return response.output_text


def rag(question: str, documents: list[str] | None = None) -> dict:
    """Run the full RAG cycle: index if needed, retrieve, and generate an answer."""
    if documents is None:
        documents = load_documents()
    collection = get_collection()
    index_corpus(collection, documents)
    contexts = retrieve(collection, question)
    answer = generate_answer(question, contexts)
    return {
        "question": question,
        "retrieved_contexts": contexts,
        "answer": answer,
    }


if __name__ == "__main__":
    docs = load_documents()
    print(f"loaded documents: {len(docs)}")

    collection = get_collection()
    index_corpus(collection, docs)
    print(f"chunks in Chroma ({CHROMA_PATH}): {collection.count()}")

    question = "What is International Atomic Time?"
    result = rag(question, documents=docs)

    print("\nquestion:", result["question"])
    print("\nretrieved_contexts:")
    for i, ctx in enumerate(result["retrieved_contexts"], 1):
        print(f"  [{i}] {ctx[:300]}{'...' if len(ctx) > 300 else ''}")
    print("\nanswer:", result["answer"])
