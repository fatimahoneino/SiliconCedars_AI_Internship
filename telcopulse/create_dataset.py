from langsmith import Client

from common.config import load_config


DATASET_NAME = "telcopulse-supervisor-evaluation"

# expected_route: the agent the supervisor should choose.
# must_contain: words the answer must include (empty = no check).
EXAMPLES = [
    ("Which three fictional subscribers have the highest churn risk scores?", "sql", ["SUB-1007"]),
    ("How many subscribers are in each country?", "sql", ["Lebanon"]),
    ("Which subscribers have more than two failed renewals?", "sql", ["SUB-1003", "SUB-1007"]),
    ("What retention offer is permitted after two failed renewals?", "rag", ["three-day"]),
    ("Which offers require manager approval?", "rag", ["manager"]),
    ("What is the monthly price of GameBox?", "rag", ["2"]),
    ("Create a bar chart of at-risk subscribers: Lebanon 2, Jordan 2, Egypt 1, UAE 1", "chart", []),
    ("Plot a pie chart of service usage: GameBox 3, GoalAlert 2, MusicDaily 1", "chart", []),
    ("Research current mobile VAS retention practices in MENA.", "research", []),
    ("Draft a friendly Arabic message offering a three-day GameBox pass.", "conversation", []),
    ("Hello, what can you help me with?", "conversation", []),
]


def main():
    load_config()
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset already exists: {DATASET_NAME}")
        return

    client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Representative TelcoPulse requests with the expected route.",
    )
    for question, route, must_contain in EXAMPLES:
        client.create_example(
            dataset_name=DATASET_NAME,
            inputs={"question": question},
            outputs={"expected_route": route, "must_contain": must_contain},
        )
    print(f"Created dataset: {DATASET_NAME}")
    print(f"Added {len(EXAMPLES)} examples.")


if __name__ == "__main__":
    main()
