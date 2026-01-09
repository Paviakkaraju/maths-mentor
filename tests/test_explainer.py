import os
import json
from pprint import pprint
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.explainer_agent import ExplainerAgent
from agents.state import create_initial_state

load_dotenv(override=True)
model = os.getenv("MODEL")

def test_explainer():
    # 1. Initialize LLM and Agent
    llm = ChatGroq(model=model, temperature=0.5)
    explainer = ExplainerAgent(llm)

    # SCENARIO 1: Computational (The "Probability" Problem)
    print("\n" + "="*60)
    print("TEST 1: COMPUTATIONAL EXPLANATION")
    print("="*60)

    state_comp = create_initial_state(
        "A bag has 4 red and 6 black balls. Find the probability of drawing 2 red balls without replacement.",
        "session_comp_001"
    )
    state_comp.update({
        "workflow_type": "computational",
        "topic": "probability",
        "final_answer": "2/15 (or approx 0.1333)",
        "solution_steps": [
            "Calculate total ways to pick 2 balls from 10: 10C2 = 45",
            "Calculate favorable ways to pick 2 red balls from 4: 4C2 = 6",
            "Divide favorable ways by total ways: 6/45 = 2/15"
        ],
        "consolidated_knowledge": "[Source 1]: P(E) = n(E)/n(S). [Source 2]: Combination formula nCr = n! / r!(n-r)!"
    })

    res_comp = explainer.explain(state_comp)
    
    print(f"\n[EXPLANATION]:\n{res_comp['explanation']}")
    print(f"\n[STEP BREAKDOWN]:")
    for i, step in enumerate(res_comp['step_explanations']):
        print(f"{i+1}. {step}")
    print(f"\n[KEY CONCEPTS]: {res_comp['key_concepts']}")
    print(f"\n[ANALOGIES]: {res_comp['analogies']}")


    # SCENARIO 2: Conceptual (The "Bayes' Theorem" Inquiry)
    print("\n" + "="*60)
    print("TEST 2: CONCEPTUAL EXPLANATION")
    print("="*60)

    state_concept = create_initial_state(
        "Explain Bayes' Theorem and give the formula.",
        "session_concept_001"
    )
    state_concept.update({
        "workflow_type": "conceptual",
        "topic": "probability",
        "consolidated_knowledge": "[Source 1]: Bayes' Theorem: P(A|B) = [P(B|A) * P(A)] / P(B). It describes the probability of an event based on prior knowledge."
    })

    res_concept = explainer.explain(state_concept)
    
    print(f"\n[EXPLANATION]:\n{res_concept['explanation']}")
    print(f"\n[KEY CONCEPTS]: {res_concept['key_concepts']}")
    print(f"\n[ANALOGIES]: {res_concept['analogies']}")

if __name__ == "__main__":
    test_explainer()