from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from .state import MathMentorState

class ExplainerOutput(BaseModel):
    """Structured pedagogical output for the student."""
    explanation: str = Field(description="The main conversational response/lesson.")
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
        You are a JEE Math Mentor. Your job is to explain the result to a student.
        
        ### GUIDELINES:
        - Use LaTeX for all math ($...$).
        - If workflow is 'conceptual': Focus on defining the term and providing a simple example.
        - If workflow is 'computational': Walk through the verified steps with a focus on 'WHY'.
        - Analogies: Create a relatable comparison (e.g., probability like a weather forecast).
        - Key Concepts: Extract 2-3 core principles (e.g., 'Law of Total Probability').
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