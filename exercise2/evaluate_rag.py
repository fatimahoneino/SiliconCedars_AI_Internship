from langchain_groq import ChatGroq
from langsmith import Client

from common.config import load_config
from exercise2.rag import MODEL, ask, get_store


DATASET_NAME = "exercise-2-rag-evaluation"

load_config()
store = get_store()
llm = ChatGroq(model=MODEL)


def rag_target(inputs):
    question = inputs["question"]
    answer, sources = ask(question, store, llm)
    return {"answer": answer, "sources": sources}


def answer_contains_required_phrase(inputs, outputs, reference_outputs):
    answer = outputs["answer"].lower()
    required_phrase = reference_outputs["required_phrase"].lower()
    return {
        "key": "answer_contains_required_phrase",
        "score": required_phrase in answer,
    }


def correct_source(inputs, outputs, reference_outputs):
    expected_source = reference_outputs["expected_source"]
    actual_sources = outputs["sources"]

    if expected_source == "none":
        return {"key": "correct_source", "score": True}

    return {
        "key": "correct_source",
        "score": expected_source in actual_sources,
    }


def correct_refusal(inputs, outputs, reference_outputs):
    expected_source = reference_outputs["expected_source"]

    if expected_source != "none":
        return {"key": "correct_refusal", "score": True}

    answer = outputs["answer"].lower()
    refusal_phrases = [
        "don't know",
        "do not know",
        "not in the context",
        "not contain",
        "no information",
    ]
    refused = any(phrase in answer for phrase in refusal_phrases)
    return {"key": "correct_refusal", "score": refused}


def main():
    client = Client()

    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise SystemExit(
            f"Dataset '{DATASET_NAME}' does not exist. "
            "Run create_dataset.py first."
        )

    results = client.evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            answer_contains_required_phrase,
            correct_source,
            correct_refusal,
        ],
        experiment_prefix="exercise-2-rag",
        description="Evaluation of the Exercise 2 FAISS and Groq RAG system.",
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()