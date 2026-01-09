import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.verifier_agent import VerifierAgent

# Add current directory to path
sys.path.append(os.getcwd())

load_dotenv()

# Setup
model_name = os.getenv("MODEL", "meta-llama/llama-3.3-70b-versatile")
print(f"Using model: {model_name}")

try:
    llm = ChatGroq(model=model_name, temperature=0.01)
    verifier = VerifierAgent(llm, verbose=True)

    # Mock State
    # We only populate the fields required by VerifierAgent.get_system_prompt
    state = {
        "problem_text": "What is the probability of a fair six-sided dice landing on an even number?",
        "variables": ["outcomes = [1, 2, 3, 4, 5, 6]", "favorable = [2, 4, 6]"],
        "constraints": [],
        "final_answer": "0.5",
        "solution_steps": ["Calculate total outcomes: 6", "Calculate favorable outcomes (even): 3", "Probability = 3/6 = 0.5"],
        "solver_trials": 0,
        "consolidated_knowledge": ""
    }

    print("Running verification...")
    result = verifier.verify(state)
    print("\nVerification successful!")
    print("Result keys:", result.keys())
    print("Is Correct:", result.get("is_correct"))

except Exception as e:
    print("\nVerification failed with error:")
    print(e)
