import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.explainer_agent import ExplainerAgent

# Add current directory to path
sys.path.append(os.getcwd())

load_dotenv()

# Setup
model_name = os.getenv("MODEL", "meta-llama/llama-3.3-70b-versatile")
# model_name = "llama-3.3-70b-versatile"
print(f"Using model: {model_name}")

try:
    llm = ChatGroq(model=model_name, temperature=0.6)
    explainer = ExplainerAgent(llm)

    # Mock State for the Ace Problem
    state = {
        "problem_text": "What is the probability of drawing an ace from a well-shuffled deck of 52 cards?",
        "topic": "probability",
        "workflow_type": "computational",
        "final_answer": "1/13 (or approx 0.0769)",
        "solution_steps": [
            "Identify the total number of outcomes (cards in a deck): n(S) = 52",
            "Identify the number of favorable outcomes (Aces): n(E) = 4 (Ace of Spades, Hearts, Diamonds, Clubs)",
            "Apply probability formula: P(E) = n(E) / n(S)",
            "Substitute values: P(E) = 4 / 52",
            "Simplify the fraction: 1 / 13",
            "Calculate decimal: 0.0769"
        ],
        "consolidated_knowledge": "[Source 1]: Probability P(E) = n(E)/n(S). [Source 2]: A standard deck has 52 cards, 4 suits, 13 ranks."
    }

    print("Running explainer...")
    result = explainer.explain(state)
    
    print("\n" + "="*80)
    print("EXPLANATION OUTPUT")
    print("="*80)
    print(result["explanation"])
    print("\n" + "="*80)
    print("STEP EXPLANATIONS")
    for step in result.get("step_explanations", []):
        print(f"- {step}")
    print("\n" + "="*80)
    print("KEY CONCEPTS:", result.get("key_concepts"))
    print("="*80)

except Exception as e:
    print("\nExplainer failed with error:")
    print(e)
