"""Run the file as a module, not as a scripts: python -m tests.test_llm"""
from utils.llm import LLM

messages = [
    (
        "system",
        "You are a playful assistant who always replies mocking or funny responses to a user query. Reply with one sentence.",
    ),
    ("human", "Learn so that you can Earn because Learn itself has Earn"),
]
completion = LLM.invoke(messages)
print(completion.content)
print()
print(completion.usage_metadata)