import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from datetime import datetime

# Define structured output (Schema) for the router
class RouterDecision(BaseModel):
    """Decision logic for routing the math problem."""
    workflow_type: Literal["conceptual", "computational", "ambiguous"] = Field(
        description="The strategy to solve the problem."
    )
    reasoning: str = Field(description="Brief explanation of why this path was chosen.")
    plan: List[str] = Field(description="Step-by-step plan for the next agents.")

class IntentRouterAgent:
    """
    Agent that analyzes parsed math problems and routes them to the 
    appropriate workflow (Conceptual vs Computational vs Ambiguous).
    """
    def __init__(self, llm: ChatGroq):
        self.router_llm = llm.with_structured_output(RouterDecision)

    def route(self, state: Any) -> Dict[str, Any]:
        """
        Main routing logic.
        """
        start_time = datetime.now()
        
        # 1. Check if Parser already halted the process
        if not state.get("should_continue", True):
            return state

        # 2. Prepare context for the Router
        parsed_data = {
            "problem_text": state.get("problem_text"),
            "topic": state.get("topic"),
            "variables": state.get("variables"),
            "constraints": state.get("constraints"),
            "parsing_issues": state.get("parsing_issues")
        }

        # 3. Prompt the Router
        prompt = f"""You are the Intent Router for an AI Math Mentor.
        Your job is to look at the parsed math problem and decide the best solution strategy.

        PARSED DATA:
        {json.dumps(parsed_data, indent=2)}

        STRATEGIES:
        1. 'conceptual': User is asking for a definition, formula, derivation, or 'how-to'. No specific values to solve for.
        2. 'computational': User provided values or equations and wants a specific numerical or algebraic answer.
        3. 'ambiguous': The input is math-related but missing critical information (e.g., 'Find x' but no equation provided).

        Based on the data, choose the strategy and provide a high-level plan."""

        try:
            # Invoke LLM
            decision = self.router_llm.invoke(prompt)
            
            # Update State
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build the trace entry
            trace_entry = {
                "agent": "intent_router",
                "decision": decision.workflow_type,
                "reasoning": decision.reasoning,
                "plan": decision.plan,
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            return {
                "workflow_type": decision.workflow_type,
                "plan": decision.plan,
                "router_trace": [trace_entry]
            }

        except Exception as e:
            error_msg = f"Router Error: {str(e)}"
            return {
                "errors": [error_msg],
                "workflow_type": "ambiguous", # Fail-safe to HITL
                "plan": ["Trigger error handling / HITL due to router failure"]
            }

# Example Usage (for testing)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
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