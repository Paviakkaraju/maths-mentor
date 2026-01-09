
from agents.intent_router import IntentRouterAgent
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_groq import ChatGroq

MODEL = os.getenv("MODEL")

llm = ChatGroq(model=MODEL, temperature=0.3)
router = IntentRouterAgent(llm)

# Mock state from Parser
# mock_state = {
#     "should_continue": True,
#     "problem_text": "What is the formula for the variance of a binomial distribution?",
#     "topic": "probability",
#     "variables": [],
#     "constraints": []
# }

# Scenario: The student wants to find a limit but hasn't provided the function.
ambiguous_state = {
"should_continue": True,
"problem_text": "Find the limit of the function as x approaches infinity.",
"topic": "calculus",
"variables": ["x"],
"constraints": ["x -> infinity"],
"parsing_issues": ["No function expression detected (e.g., f(x) is missing)"]
}

# Scenario: A classic JEE-style probability question with specific values.
computational_state = {
"should_continue": True,
"problem_text": "A bag contains 4 red and 6 black balls. Two balls are drawn at random without replacement. Find the probability that both are the same color.",
"topic": "probability",
"variables": ["red_balls=4", "black_balls=6", "draws=2"],
"constraints": ["without replacement", "same color"],
"parsing_issues": []
}
# result = router.route(mock_state)
# print(json.dumps(result, indent=2))

print("\n--- Testing Ambiguous Input ---")
result_ambiguous = router.route(ambiguous_state)
print(f"Decision: {result_ambiguous['workflow_type']}")
print(f"Plan: {result_ambiguous['plan']}")

print("\n--- Testing Computational Problem ---")
result_comp = router.route(computational_state)
print(f"Decision: {result_comp['workflow_type']}")
print(f"Plan: {result_comp['plan']}")