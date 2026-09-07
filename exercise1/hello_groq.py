from langchain_groq import ChatGroq

from common.config import load_config


load_config()
llm = ChatGroq(model="openai/gpt-oss-20b")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. "
        "Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)