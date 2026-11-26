from langchain_groq import ChatGroq
from langsmith import Client

from common.config import load_config
from exercise4.agent import MODEL, run_agent
from exercise4.create_dataset import DATASET_NAME
from exercise4.tools import ALL_TOOLS, init_db


load_config()
init_db()
llm = ChatGroq(model=MODEL).bind_tools(ALL_TOOLS)


def agent_target(inputs):
    answer, used_tools = run_agent(inputs["question"], llm)
    return {"answer": answer, "used_tools": used_tools}


def correct_tool_sequence(inputs, outputs, reference_outputs):
    """Did the agent use the expected tools, in the expected order?"""
    expected = reference_outputs["expected_tools"]
    actual = list(dict.fromkeys(outputs["used_tools"]))  # remove repeats, keep order
    return {
        "key": "correct_tool_sequence",
        "score": actual == expected,
        "comment": f"expected={expected} actual={actual}",
    }


def no_repeated_calls(inputs, outputs, reference_outputs):
    """Did the agent avoid calling the same tool twice?"""
    used = outputs["used_tools"]
    return {
        "key": "no_repeated_calls",
        "score": len(used) == len(set(used)),
        "comment": f"calls={used}",
    }


def main():
    client = Client()

    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise SystemExit(
            f"Dataset '{DATASET_NAME}' does not exist. "
            "Run python -m exercise4.create_dataset first."
        )

    results = client.evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[correct_tool_sequence, no_repeated_calls],
        experiment_prefix="exercise-4-agent",
        description="Tool selection evaluation for the Exercise 4 agent.",
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()
