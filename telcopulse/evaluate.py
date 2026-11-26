import uuid

from langsmith import Client

from common.config import load_config
from telcopulse.create_dataset import DATASET_NAME
from telcopulse.database import init_db
from telcopulse.supervisor import ask, build_app


load_config()
init_db()
app = build_app()


def supervisor_target(inputs):
    # A fresh thread for each question, so answers do not influence each other.
    result = ask(app, inputs["question"], thread_id=str(uuid.uuid4()))
    return {
        "route": result["route"],
        "answer": result["answer"],
        "has_chart": bool(result.get("chart")),
        "retries": result.get("retries", 0),
        "completed": not result.get("failed", False),
        "elapsed_ms": result["elapsed_ms"],
    }


def correct_route(inputs, outputs, reference_outputs):
    expected = reference_outputs["expected_route"]
    return {
        "key": "correct_route",
        "score": outputs["route"] == expected,
        "comment": f"expected={expected} actual={outputs['route']}",
    }


def completed(inputs, outputs, reference_outputs):
    return {"key": "completed", "score": outputs["completed"]}


def answer_contains_facts(inputs, outputs, reference_outputs):
    words = reference_outputs.get("must_contain", [])
    answer = outputs["answer"].lower().replace("\u2011", "-")
    missing = [word for word in words if word.lower() not in answer]
    return {
        "key": "answer_contains_facts",
        "score": not missing,
        "comment": f"missing={missing}" if missing else "ok",
    }


def chart_created(inputs, outputs, reference_outputs):
    if reference_outputs["expected_route"] != "chart":
        return {"key": "chart_created", "score": True}
    return {"key": "chart_created", "score": outputs["has_chart"]}


def no_fallback(inputs, outputs, reference_outputs):
    return {
        "key": "no_fallback",
        "score": outputs["retries"] == 0,
        "comment": f"retries={outputs['retries']}",
    }


def main():
    client = Client()

    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise SystemExit(
            f"Dataset '{DATASET_NAME}' does not exist. "
            "Run python -m telcopulse.create_dataset first."
        )

    results = client.evaluate(
        supervisor_target,
        data=DATASET_NAME,
        evaluators=[correct_route, completed, answer_contains_facts, chart_created, no_fallback],
        experiment_prefix="telcopulse-supervisor",
        description="Routing, completion, facts and fallback checks for TelcoPulse.",
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()
