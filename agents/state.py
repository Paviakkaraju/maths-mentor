"""
State Schema for ReAct Multi-Agent Math Mentor System.
This defines the shared state that flows through all agents.
"""

from typing import TypedDict, List, Dict, Optional, Annotated, Literal
from datetime import datetime
import operator

class MathMentorState(TypedDict):
    """
    Complete state for math problem solving with ReAct agents.
    """
    
    # Input
    session_id: str
    raw_input: str
    input_mode: Literal["text", "image", "audio"]
    timestamp: str
    
    # Parser Agent Outputs
    # Intent classification
    intent: Optional[Literal["math_problem", "chitchat", "out_of_context"]]
    intent_confidence: float
    
    # If chitchat or OOC, parser responds directly
    direct_response: Optional[str]  # "Hi! I'm a math tutor..."
    should_continue: bool  # False if chitchat/OOC, True if math
    
    # If math problem, parsing results
    problem_text: str  # Cleaned, structured problem
    topic: Optional[Literal["probability", "algebra", "calculus", "linear_algebra", "unknown"]]
    variables: List[str]
    constraints: List[str]
    parsing_valid: bool
    parsing_issues: List[str]
    
    # Solver Agent Outputs
    # Knowledge retrieval
    rag_chunks: List[Dict]  # Retrieved from knowledge base
    similar_problems: List[Dict]  # Retrieved from memory
    
    # Solution
    solution_steps: List[str]
    final_answer: str
    solver_reasoning: str  # Why this approach was chosen
    calculations_performed: List[Dict]  # [{code: "...", result: "..."}]
    tools_used: List[str]  # Which tools were actually used
    
    # Verifier Agent Outputs
    is_correct: Optional[bool]  # True/False/None
    verification_confidence: float
    verification_notes: str
    constraints_satisfied: bool
    constraints_violations: List[str]
    math_steps_valid: bool
    verification_tools_used: List[str]
    
    # Explainer Agent Outputs
    explanation: str  # Full pedagogical explanation
    step_explanations: List[str]  # Each step explained
    key_concepts: List[str]
    analogies: List[str]
    
    # Agent Reasoning Traces
    parser_trace: Annotated[List[Dict], operator.add]
    solver_trace: Annotated[List[Dict], operator.add]
    verifier_trace: Annotated[List[Dict], operator.add]
    explainer_trace: Annotated[List[Dict], operator.add]
    
    # Errors and Meta-data
    errors: Annotated[List[str], operator.add]
    execution_time_ms: Optional[float]


def create_initial_state(
    user_input: str,
    session_id: str,
    input_mode: Literal["text", "image", "audio"] = "text"
) -> MathMentorState:
    """
    Function to create initial state.
    
    Args:
        user_input: Raw user input
        session_id: Unique session identifier
        input_mode: Type of input
    
    Returns:
        Initialized MathMentorState
    """
    return MathMentorState(
        # Input
        session_id=session_id,
        raw_input=user_input,
        input_mode=input_mode,
        timestamp=datetime.now().isoformat(),
        
        # Parser defaults
        intent=None,
        intent_confidence=0.0,
        direct_response=None,
        should_continue=True,
        problem_text="",
        topic=None,
        variables=[],
        constraints=[],
        parsing_valid=False,
        parsing_issues=[],
        
        # Solver defaults
        rag_chunks=[],
        similar_problems=[],
        solution_steps=[],
        final_answer="",
        solver_reasoning="",
        calculations_performed=[],
        tools_used=[],
        
        # Verifier defaults
        is_correct=None,
        verification_confidence=0.0,
        verification_notes="",
        constraints_satisfied=True,
        constraints_violations=[],
        math_steps_valid=False,
        verification_tools_used=[],
        
        # Explainer defaults
        explanation="",
        step_explanations=[],
        key_concepts=[],
        analogies=[],
        
        # Traces
        parser_trace=[],
        solver_trace=[],
        verifier_trace=[],
        explainer_trace=[],
        
        # Errors
        errors=[],
        execution_time_ms=None
    )