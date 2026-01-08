from typing import Dict, Any, List, Literal, Optional
from datetime import datetime
import json
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from .state import MathMentorState

# --- Optimized Output Schema ---
class ParserOutput(BaseModel):
    """The final structured output from the Parser Agent."""
    intent: Literal["math_problem", "chitchat", "out_of_context"]
    confidence: float
    
    # Use 'unknown' as the mandatory fallback for non-math topics
    topic: Literal["probability", "algebra", "calculus", "linear_algebra", "unknown"] = Field(
        default="unknown"
    )
    
    # Allow these to be empty lists by default
    problem_text: str = Field(default="")
    variables: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    
    is_valid: bool = Field(default=True)
    issues: List[str] = Field(default_factory=list)
    direct_response: Optional[str] = Field(default=None)

class ParserAgent:
    """
    A streamlined Parser Agent.
    Instead of a complex ReAct loop, it uses high-accuracy structured output
    to perform intent classification, parsing, and validation in one shot.
    """
    def __init__(self, llm: ChatGroq):
        # Bind the structured output to the LLM
        self.parser_chain = llm.with_structured_output(ParserOutput)

    def parse(self, state: MathMentorState) -> MathMentorState:
        raw_input = state["raw_input"]
        start_time = datetime.now()

        prompt = f"""You are an expert Math Parser for a JEE-level tutoring system.
        
        USER INPUT: "{raw_input}"

        TASK:
        1. Classify Intent: Is this a math_problem, chitchat, or out_of_context?
        2. Parse: If it's a math problem, extract topic, variables, and semantic constraints (e.g., 'without replacement').
        3. Validate: Is the problem solvable? (e.g., 'Find x' without an equation is invalid).
        4.If NOT math_problem: 
            - Set topic to "unknown"
            - Set variables and constraints to []
            - Provide a friendly 'direct_response' that you can only resolve queries related Math Problems 
            and especially in probability, algebra, calculus, linear_algebra.

        GUIDELINES:
        - Preserve LaTeX notation ($x^2$).
        - For probability, look specifically for sampling constraints.

        CRITICAL:
        - 'topic' MUST be one of: "probability", "algebra", "calculus", "linear_algebra", or "unknown". 
        - Do NOT leave 'topic' as an empty string. If it is not math, use "unknown".
        - 'confidence' must be a JSON number (e.g. 0.9), not a string.
        """

        try:
            # Single-pass execution
            result = self.parser_chain.invoke(prompt)

            # Update State
            state["intent"] = result.intent
            state["intent_confidence"] = result.confidence
            state["problem_text"] = result.problem_text
            state["topic"] = result.topic
            state["variables"] = result.variables
            state["constraints"] = result.constraints
            state["parsing_valid"] = result.is_valid
            state["parsing_issues"] = result.issues
            state["direct_response"] = result.direct_response
            
            # Logic for continuing the graph
            if result.intent != "math_problem" or not result.is_valid:
                state["should_continue"] = False
            else:
                state["should_continue"] = True

            # Add trace for UI requirement
            state["parser_trace"].append({
                "agent": "parser",
                "intent_detected": result.intent,
                "is_valid": result.is_valid,
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            })

        except Exception as e:
            # Handle the specific Groq 400 error if it persists
            error_msg = str(e)
            if "expected number, but got string" in error_msg:
                state["errors"].append("Groq Type Error: LLM failed to output a raw number. Try re-prompting.")
            else:
                state["errors"].append(f"Parser Error: {error_msg}")
            state["should_continue"] = False

        return state