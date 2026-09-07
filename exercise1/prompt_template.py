from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from common.config import load_config


load_config()
llm = ChatGroq(model="openai/gpt-oss-20b")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates English to {language}."),
        ("human", "{text}"),
    ]
)

filled = prompt.invoke({"language": "japanese", "text": "I love you."})
ai_msg = llm.invoke(filled)
print(ai_msg.content)