from langsmith import Client

from common.config import load_config


DATASET_NAME = "exercise-4-tool-selection"

# Each question lists the tools the agent should use, in order.
EXAMPLES = [
    ("Translate 'good evening' to Japanese.", ["translate_to_japanese"]),
    ("What is the capital and population of Okinawa prefecture?", ["lookup_prefecture"]),
    ("Search the web for the latest cherry blossom forecast in Japan.", ["search_japan"]),
    (
        "Translate 'thank you very much' to Japanese, then give me Hokkaido's capital.",
        ["translate_to_japanese", "lookup_prefecture"],
    ),
    ("Tell me a one-line joke about sushi.", []),
]


def main():
    load_config()
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset already exists: {DATASET_NAME}")
        return

    client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Expected tool choices for the Exercise 4 Japan agent.",
    )
    for question, expected_tools in EXAMPLES:
        client.create_example(
            dataset_name=DATASET_NAME,
            inputs={"question": question},
            outputs={"expected_tools": expected_tools},
        )
    print(f"Created dataset: {DATASET_NAME}")
    print(f"Added {len(EXAMPLES)} examples.")


if __name__ == "__main__":
    main()
