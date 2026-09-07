import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from common.config import load_config


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates English to {language}."),
        ("human", "{text}"),
    ]
)


def models_comparator(model_names, prompt_values):
    """Compare up to five models and list them from fastest to slowest."""
    if not 1 <= len(model_names) <= 5:
        raise ValueError("Choose between 1 and 5 models.")

    load_config()
    filled_prompt = PROMPT.invoke(prompt_values)
    results = []

    for name in model_names:
        llm = ChatGroq(model=name)
        start = time.time()
        ai_msg = llm.invoke(filled_prompt)
        elapsed_ms = round((time.time() - start) * 1000, 2)
        results.append((name, ai_msg.content, elapsed_ms))

    return sorted(results, key=lambda result: result[2])


if __name__ == "__main__":
    results = models_comparator(
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        {"language": "french", "text": "I love you."},
    )

    for model_name, answer, elapsed_ms in results:
        print(f"\n{model_name} ({elapsed_ms} ms)")
        print(answer)