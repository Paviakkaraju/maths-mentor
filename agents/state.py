from typing import TypedDict, List, Dict, Optional, Annotated, Literal
from datetime import datetime
import operator

class MathMentorState(TypedDict):
    """
    Updated State Schema for ReAct Multi-Agent Math Mentor System.
    """
    # Input
    session_id: str
    raw_input: str
    input_mode: Literal["text", "image", "audio"]
    timestamp: str
    
    # Parser Agent Outputs
    intent: Optional[Literal["math_problem", "chitchat", "out_of_context"]]
    intent_confidence: float
    direct_response: Optional[str] 
    should_continue: bool  
    problem_text: str  
    topic: Optional[Literal["probability", "algebra", "calculus", "linear_algebra", "unknown"]]
    variables: List[str]
    constraints: List[str]
    parsing_valid: bool
    parsing_issues: List[str]

    # Intent Router Outputs ---
    workflow_type: Optional[Literal["conceptual", "computational", "ambiguous"]]
    plan: List[str] # Sequential steps for the agents to follow
    
    # Solver Agent Outputs
    rag_chunks: List[Dict]  
    similar_problems: List[Dict]  
    solution_steps: List[str]
    final_answer: str
    solver_reasoning: str  
    calculations_performed: List[Dict]  
    consolidated_knowledge: str  # The "Cheat Sheet" for the Solver or the raw material for the Explainer
    tools_used: List[str]  
    
    # Verifier Agent Outputs
    is_correct: Optional[bool]  
    verification_confidence: float
    verification_notes: str
    constraints_satisfied: bool
    constraints_violations: List[str]
    math_steps_valid: bool
    verification_tools_used: List[str]
    
    # Explainer Agent Outputs
    explanation: str  
    step_explanations: List[str]  
    key_concepts: List[str]
    analogies: List[str]
    
    # Agent Reasoning Traces
    parser_trace: Annotated[List[Dict], operator.add]
    router_trace: Annotated[List[Dict], operator.add] # NEW
    rag_trace: Annotated[List[Dict], operator.add]  
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
    return MathMentorState(
        session_id=session_id,
        raw_input=user_input,
        input_mode=input_mode,
        timestamp=datetime.now().isoformat(),
        
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

        
        workflow_type=None,
        plan=[],
        
        rag_chunks=[],
        similar_problems=[],
        solution_steps=[],
        final_answer="",
        solver_reasoning="",
        calculations_performed=[],
        tools_used=[],
        
        is_correct=None,
        verification_confidence=0.0,
        verification_notes="",
        constraints_satisfied=True,
        constraints_violations=[],
        math_steps_valid=False,
        verification_tools_used=[],
        
        explanation="",
        step_explanations=[],
        key_concepts=[],
        analogies=[],
        
        parser_trace=[],
        router_trace=[], 
        rag_trace=[],
        solver_trace=[],
        verifier_trace=[],
        explainer_trace=[],
        
        errors=[],
        execution_time_ms=None
    )