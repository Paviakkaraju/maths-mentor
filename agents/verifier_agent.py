import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from .hardened_repl import HardenedMathREPL
from .state import MathMentorState

class VerifierAgent:
    """
    Agent that verifies the Solver's output.
    Uses Independent Verification (Reverse Solving) via Hardened REPL.
    """
    def __init__(self, llm, max_iterations: int = 3, verbose: bool = False):
        self.llm = llm
        self.repl = HardenedMathREPL()
        self.max_iterations = max_iterations
        self.verbose = verbose

    def get_system_prompt(self, state: MathMentorState) -> str:
        problem = state.get('problem_text', '')
        variables = state.get('variables', [])
        constraints = state.get('constraints', [])
        solver_answer = state.get('final_answer', '')
        solver_steps = state.get('solution_steps', [])
        trials = state.get('solver_trials', 0)

        return f"""You are a Senior Math Verifier. Your role is to peer-review a Solver's work.
        
### CURRENT ATTEMPT: {trials + 1}

### AUDIT CONTEXT
- PROBLEM: {problem}
- VARIABLES: {variables}
- CONSTRAINTS: {constraints}
- SOLVER'S ANSWER: {solver_answer}
- SOLVER'S LOGIC: {solver_steps}

### YOUR MISSION
Perform an independent verification. Do NOT just repeat the solver's steps. 
Instead, try to:
1. **Reverse Check**: If the answer is X, does it satisfy the original equations?
2. **Alternative Path**: Is there a simpler way to calculate this to confirm the result?
3. **Constraint Check**: Does the answer violate "without replacement", "non-negative", etc.?

### TOOLS
You MUST use the `python_verifier_repl` to execute your independent check.

### OUTPUT FORMAT
1. **PLAN**: State how you will independently verify the answer.
2. **CODE**: Write the Python code for verification.
3. **FINAL DECISION**: 
   - If correct: State "VERIFIED: The solution is logically sound and mathematically correct."
   - If incorrect: State "REJECTED: [Reason]" and provide specific feedback for the solver.
"""

    def verify(self, state: MathMentorState) -> Dict[str, Any]:
        start_time = datetime.now()
        
        @tool
        def python_verifier_repl(code: str) -> str:
            """Execute Python for verification. Pre-imported: math, statistics, sympy (sp), numpy (np)."""
            return self.repl.execute(code)

        llm_with_tools = self.llm.bind_tools([python_verifier_repl])
        
        messages = [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content="Review the solver's work and provide a final decision.")
        ]

        structured_trace = []
        is_correct = False
        feedback = ""

        # ReAct Loop for Verification
        for i in range(self.max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # Check for the "VERIFIED" keyword in final response
                if "VERIFIED" in response.content.upper():
                    is_correct = True
                else:
                    is_correct = False
                    feedback = response.content
                break

            for tool_call in response.tool_calls:
                # Execute verification code
                result = python_verifier_repl.invoke(tool_call["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
                
                structured_trace.append({
                    "step": f"Verification Step {i+1}",
                    "code": tool_call["args"].get("code", ""),
                    "result": result
                })

        # --- LOGIC: Handle the feedback loop ---
        updates = {
            "is_correct": is_correct,
            "verification_notes": feedback if not is_correct else "Solution verified successfully.",
            "verifier_trace": structured_trace,
            "math_steps_valid": is_correct,
            "verification_tools_used": ["hardened_math_repl"]
        }

        if not is_correct:
            current_trials = state.get("solver_trials", 0)
            if current_trials == 0:
                # FIRST FAILURE: Route back to Solver
                updates["solver_trials"] = 1
                current_kb = state.get("consolidated_knowledge", "")
                
                # Append the "Mentor Note" to the Cheat Sheet
                updates["consolidated_knowledge"] = current_kb + f"""
                
            ### MENTOR FEEDBACK (ATTEMPT 1 FAILED) ###
            The previous solution was rejected.
            REASON: {feedback}
            Please re-evaluate the problem, focusing on the constraints and the feedback above.
            ##########################################
"""
            else:
                # SECOND FAILURE: Halt (In future, this goes to HITL)
                updates["should_continue"] = False 
                updates["verification_notes"] += "\n[SYSTEM]: Max retries reached. Manual intervention suggested."

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        updates["execution_time_ms"] = execution_time

        return updates