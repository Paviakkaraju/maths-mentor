# """
# Test script for ParserAgent

# Run:
#     python test_parser_agent.py
# """

import os
from pprint import pprint
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from agents.parser_agent import ParserAgent
from agents.state import MathMentorState, create_initial_state

load_dotenv()

def make_state(user_input: str) -> MathMentorState:
    """Create a fresh MathMentorState for testing."""
    return {
        "raw_input": user_input,
        "intent": None,
        "intent_confidence": 0.0,
        "problem_text": "",
        "topic": "unknown",
        "variables": [],
        "constraints": [],
        "parsing_valid": False,
        "parsing_issues": [],
        "direct_response": "",
        "should_continue": True,
        "errors": [],
        "parser_trace": [],
    }


# def run_test_case(agent: ParserAgent, text: str):
#     print("\n" + "=" * 70)
#     print(f"USER INPUT: {text}")
#     print("=" * 70)

#     state = make_state(text)
#     updated = agent.parse(state)

#     print("\n--- RESULT STATE ---")
#     pprint(updated)

#     print("\n--- TRACE (ReAct steps) ---")
#     for t in updated["parser_trace"]:
#         pprint(t)


# if __name__ == "__main__":

#     if not os.getenv("GROQ_API_KEY"):
#         raise RuntimeError("Please set GROQ_API_KEY in your environment.")

#     llm = ChatGroq(
#         model="meta-llama/llama-4-scout-17b-16e-instruct",
#         temperature=0.3
#     )

#     agent = ParserAgent(llm)

#     # ---- TEST CASES ----

#     # Math question
#     run_test_case(agent, "If x + 3 = 10, what is the value of x?")

#     # Chit-chat
#     run_test_case(agent, "Hey, how are you?")

#     # Out-of-context
#     run_test_case(agent, "What is the weather in Bangalore?")

#     # More complex math
#     run_test_case(agent, "Find the probability that a fair die shows an even number.")

# Initialize
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
parser_agent = ParserAgent(llm)

# Run
alg = "If x + 3 = 10, what is x?"
chat = "Hello! Good Morning!"
ooc = "What is the boiling temperature of water?"
prob = "A bag contains 4 red and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both are the same color."

state = make_state(alg)
state = parser_agent.parse(state)

print(f"Intent: {state['intent']}")
print(f"Should Continue: {state['should_continue']}")
print(f"Topic: {state['topic']}")
print(f"Variables: {state['variables']}")
print(f"Constraints: {state['constraints']}")
print(f"Response: {state["direct_response"]}")
print(f"Errors: {state["errors"]}")