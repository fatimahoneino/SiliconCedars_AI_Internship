from langsmith import Client

from common.config import load_config


DATASET_NAME = "exercise-2-rag-evaluation"

EXAMPLES = [
    {
        "inputs": {"question": "How many days of paid leave do employees receive?"},
        "outputs": {
            "expected_answer": "Employees receive 25 days of paid leave per year.",
            "expected_source": "company_policies.txt",
            "required_phrase": "25 days",
        },
    },
    {
        "inputs": {
            "question": "How many leave days can employees carry into the next year?"
        },
        "outputs": {
            "expected_answer": "Employees can carry a maximum of 10 days into the next year.",
            "expected_source": "company_policies.txt",
            "required_phrase": "10 days",
        },
    },
    {
        "inputs": {"question": "What is the refund policy for annual plans?"},
        "outputs": {
            "expected_answer": (
                "Annual plans are refunded pro rata when cancellation happens "
                "more than three months before renewal."
            ),
            "expected_source": "product_faq.txt",
            "required_phrase": "three months",
        },
    },
    {
        "inputs": {"question": "How quickly should pull requests be reviewed?"},
        "outputs": {
            "expected_answer": "Reviews are expected within one working day.",
            "expected_source": "engineering_handbook.txt",
            "required_phrase": "one working day",
        },
    },
    {
        "inputs": {"question": "What is the capital of Peru?"},
        "outputs": {
            "expected_answer": "The supplied documents do not contain this information.",
            "expected_source": "none",
            "required_phrase": "don't know",
        },
    },
]


def main():
    load_config()
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset already exists: {DATASET_NAME}")
        print("Nothing was added.")
        return

    client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Questions used to evaluate the Exercise 2 RAG system. "
            "Includes document questions and one out-of-scope question."
        ),
    )

    for example in EXAMPLES:
        client.create_example(
            dataset_name=DATASET_NAME,
            inputs=example["inputs"],
            outputs=example["outputs"],
        )

    print(f"Created dataset: {DATASET_NAME}")
    print(f"Added {len(EXAMPLES)} examples.")


if __name__ == "__main__":
    main()