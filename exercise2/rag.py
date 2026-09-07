import sys
import time
from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import load_config

MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

DOCS_DIR = Path(__file__).parent / "documents"
INDEX_DIR = Path(__file__).parent / "faiss_index"

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def read_documents():
    documents = []

    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        document = Document(
            page_content=text,
            metadata={"source": path.name},
        )
        documents.append(document)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_index(chunks, embeddings):
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    return store


def load_index(embeddings):
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_store(rebuild=False):
    embeddings = FastEmbedEmbeddings()
    index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "index.pkl"

    if index_file.exists() and metadata_file.exists() and not rebuild:
        print("Loading the existing index")
        return load_index(embeddings)

    documents = read_documents()
    chunks = split_documents(documents)
    print(f"Indexing {len(chunks)} chunks")
    return build_index(chunks, embeddings)


def format_context(chunks):
    parts = []

    for chunk in chunks:
        source = chunk.metadata["source"]
        parts.append(f"[{source}]\n{chunk.page_content}")

    return "\n\n".join(parts)


def ask(question, store, llm):
    retrieved = store.similarity_search(question, k=TOP_K)

    if not retrieved:
        return "No relevant information was found.", []

    filled_prompt = ANSWER_PROMPT.invoke(
        {
            "context": format_context(retrieved),
            "question": question,
        }
    )
    reply = llm.invoke(filled_prompt)
    sources = sorted({chunk.metadata["source"] for chunk in retrieved})
    return reply.content, sources


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_config()
    store = get_store()
    llm = ChatGroq(model=MODEL)

    questions = [
        "How many days of paid leave do employees receive, and how many can they carry over?",
        "What is the refund policy for annual plans?",
        "How quickly should pull requests be reviewed?",
        "What is the capital of Peru?",
    ]

    for question in questions:
        start = time.time()
        answer, sources = ask(question, store, llm)
        elapsed_ms = round((time.time() - start) * 1000, 2)

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"Sources: {', '.join(sources) if sources else 'none'}")
        print(f"Time: {elapsed_ms} ms")


if __name__ == "__main__":
    main()