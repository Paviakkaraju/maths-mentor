# from typing import Dict, Any, List
# from langchain_core.messages import HumanMessage, SystemMessage
# from .state import MathMentorState
# from datetime import datetime

# class ExplainerAgent:
#     """
#     The Pedagogical Layer. Handles both conceptual explanations 
#     and step-by-step solution walkthroughs.
#     """
#     def __init__(self, llm):
#         self.llm = llm

#     def explain(self, state: MathMentorState) -> Dict[str, Any]:
#         workflow = state.get("workflow_type", "computational")
        
#         # 1. Handle Dynamic Input based on Workflow
#         if workflow == "conceptual":
#             # Focus on RAG data only
#             lesson_context = f"""
#             CONCEPTUAL QUERY: {state['problem_text']}
#             RETRIEVED KNOWLEDGE: {state.get('consolidated_knowledge', 'No data found.')}
#             """
#             system_instruction = """
#             You are a Math Mentor explaining a concept. 
#             The student is asking for a definition, formula, or theory.
            
#             STRUCTURE:
#             1. The Concept: Define it clearly using LaTeX ($...$).
#             2. Why it Matters: Explain its importance in JEE math.
#             3. Example: Provide a simple illustrative example.
#             4. Mentor's Tip: How to remember this or a common pitfall.
#             """
#         else:
#             # Focus on the Verified Solution
#             lesson_context = f"""
#             PROBLEM: {state['problem_text']}
#             VERIFIED ANSWER: {state.get('final_answer', 'N/A')}
#             SOLUTION STEPS: {state.get('solution_steps', [])}
#             KNOWLEDGE USED: {state.get('consolidated_knowledge', 'N/A')}
#             """
#             system_instruction = """
#             You are a Math Mentor explaining a solved problem.
            
#             STRUCTURE:
#             1. Summary: Confirm the final answer warmly.
#             2. The Logic: Walk through the solution steps pedagogically. Explain 'WHY' we took each step.
#             3. Key Formula: Highlight the main formula used from the knowledge base.
#             4. Common Pitfall: Warn about mistakes often made in this specific problem type.
#             """

#         # 2. Generate the Pedagogical Response
#         response = self.llm.invoke([
#             SystemMessage(content=system_instruction),
#             HumanMessage(content=f"Provide a student-friendly explanation for this context:\n{lesson_context}")
#         ])

#         return {
#             "explanation": response.content,
#             "explainer_trace": [{
#                 "event": "explanation_complete",
#                 "workflow_type": workflow,
#                 "timestamp": datetime.now().isoformat()
#             }]
#         }

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