from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from .state import MathMentorState

class ExplainerOutput(BaseModel):
    """Structured pedagogical output for the student."""
    explanation: str = Field(description="The main lesson content.")
    step_explanations: List[str] = Field(description="Breakdown of the solution logic into friendly steps.")
    key_concepts: List[str] = Field(description="The core mathematical principles involved.")
    analogies: List[str] = Field(description="Real-world analogies to help intuition.")

class ExplainerAgent:
    """
    The final pedagogical layer. 
    Converts raw math facts into a structured, student-friendly lesson.
    """
    def __init__(self, llm):
        # We use structured output to ensure state fields are always populated
        self.llm = llm.with_structured_output(ExplainerOutput)

    def explain(self, state: MathMentorState) -> Dict[str, Any]:
        workflow = state.get("workflow_type", "computational")
        
        # 1. Dynamic Context Preparation
        if workflow == "conceptual":
            context = f"""
            WORKFLOW: Conceptual Inquiry
            TOPIC: {state.get('topic')}
            QUERY: {state['problem_text']}
            KNOWLEDGE BASE: {state.get('consolidated_knowledge')}
            """
        else:
            context = f"""
            WORKFLOW: Computational Problem
            PROBLEM: {state['problem_text']}
            VERIFIED ANSWER: {state.get('final_answer')}
            SOLVER STEPS: {state.get('solution_steps')}
            KNOWLEDGE USED: {state.get('consolidated_knowledge')}
            """

        # 2. The Mentor System Prompt
        system_prompt = f"""
        You are a JEE Math Mentor. Explain the solution to the student.
        
        ### OUTPUT FIELDS INSTRUCTIONS:
        
        1. **explanation** (String):
           - Provide a brief, encouraging introduction.
           - State the goal of the problem.
           - Keep this short (1-2 sentences).
           
        2. **step_explanations** (List[String]):
           - This is the CRITICAL part.
           - Convert the 'SOLVER STEPS' into a clear, detailed list.
           - Each item in the list must be a self-contained step.
           - Use LaTeX ($...$) for formulas and calculations in every step.
           - Example item: "First, we identify the total outcomes. Since it is a deck of cards, $n(S) = 52$."
           
        3. **key_concepts** (List[String]):
           - Extract 2-3 core formulas or rules used.
           
        ### RULES:
        - Use LaTeX ($...$) for all math.
        - Be educational.
        """

        # 3. Invoke LLM
        output = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Explain the following context to the student:\n{context}")
        ])

        # 4. Return Updates to Global State
        return {
            "explanation": output.explanation,
            "step_explanations": output.step_explanations,
            "key_concepts": output.key_concepts,
            "analogies": output.analogies,
            "explainer_trace": [{
                "event": "pedagogy_complete",
                "workflow": workflow,
                "timestamp": datetime.now().isoformat()
            }]
        }