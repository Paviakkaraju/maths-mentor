import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.verifier_agent import VerifierAgent
from agents.state import create_initial_state
from utils.llm import LLM

load_dotenv()

def run_verifier_test():
    # 1. Initialize LLM and Verifier
     # llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    verifier = VerifierAgent(LLM, verbose=True)

    # ---------------------------------------------------------
    # SCENARIO 1: Solver is CORRECT
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 1: CORRECT SOLVER OUTPUT")
    print("="*60)
    
    state_correct = create_initial_state(
        "A bag has 4 red and 6 black balls. Probability of 2 red without replacement?", 
        "test_001"
    )
    state_correct.update({
        "problem_text": "A bag has 4 red and 6 black balls. Probability of 2 red without replacement?",
        "topic": "probability",
        "final_answer": "0.1333",
        "solution_steps": ["1. Total ways = 10C2 = 45", "2. Red ways = 4C2 = 6", "3. Prob = 6/45 = 0.1333"],
        "solver_trials": 0,
        "consolidated_knowledge": "[Source 1]: P(E) = n(E)/n(S). [Source 2]: nCr = n! / r!(n-r)!"
    })

    result_1 = verifier.verify(state_correct)
    print(f"Is Correct: {result_1['is_correct']}")
    print(f"Notes: {result_1['verification_notes']}")

    # ---------------------------------------------------------
    # SCENARIO 2: Solver is WRONG (Confidently Hallucinating)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2: WRONG SOLVER OUTPUT (Logic Error)")
    print("="*60)
    
    state_wrong = create_initial_state(
        "A bag has 4 red and 6 black balls. Probability of 2 red without replacement?", 
        "test_002"
    )
    # Mocking a common mistake: using "with replacement" logic (10*10 instead of 10*9)
    state_wrong.update({
        "problem_text": "A bag has 4 red and 6 black balls. Probability of 2 red without replacement?",
        "final_answer": "0.16", 
        "solution_steps": ["1. Total ways = 10*10 = 100", "2. Red ways = 4*4 = 16", "3. Prob = 16/100 = 0.16"],
        "solver_trials": 0,
        "consolidated_knowledge": "[Source 1]: P(E) = n(E)/n(S)."
    })

    result_2 = verifier.verify(state_wrong)
    print(f"Is Correct: {result_2['is_correct']}")
    print(f"Solver Trials Updated to: {result_2.get('solver_trials')}")
    print(f"Feedback injected into Knowledge: {'MENTOR FEEDBACK' in result_2.get('consolidated_knowledge', '')}")
    print(f"Notes: {result_2['verification_notes']}")

if __name__ == "__main__":
    run_verifier_test()