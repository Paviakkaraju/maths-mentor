# from typing import Dict, Any, List, Literal, Optional
# from datetime import datetime
# import json
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage
# from .state import MathMentorState

# # Parser Output Schema 
# class ParserOutput(BaseModel):
#     """The final structured output from the Parser Agent."""
#     intent: Literal["math_problem", "chitchat", "out_of_context"]
#     confidence: float

#     topic: Literal["probability", "algebra", "calculus", "linear_algebra", "unknown"] = Field(
#         default="unknown"
#     )
    
#     # Allow these to be empty lists by default
#     problem_text: str = Field(default="")
#     variables: List[str] = Field(default_factory=list)
#     constraints: List[str] = Field(default_factory=list)
    
#     is_valid: bool = Field(default=True)
#     issues: List[str] = Field(default_factory=list)
#     direct_response: Optional[str] = Field(default=None)

# class ParserAgent:
#     """
#     A streamlined Parser Agent.
#     Instead of a complex ReAct loop, it uses high-accuracy structured output
#     to perform intent classification, parsing, and validation in one shot.
#     """
#     def __init__(self, llm: ChatGroq):
#         # Bind the structured output to the LLM
#         self.parser_chain = llm.with_structured_output(ParserOutput)

#     def parse(self, state: MathMentorState) -> MathMentorState:
#         raw_input = state["raw_input"]
#         start_time = datetime.now()

#         prompt = f"""You are an expert Math Parser for a JEE-level tutoring system.
        
# USER INPUT: "{raw_input}"

# TASK:
# 1. Classify Intent: Is this a math_problem, chitchat, or out_of_context?
#    - "math_problem": ANY query related to math. This includes numerical problems (e.g., "Solve x+2=5") AND conceptual/theoretical questions (e.g., "Explain Bayes' Theorem", "What is a derivative?", "Give me the formula for variance").
#    - "chitchat": Greetings (Hi, Hello), pleasantries (How are you?), or thanks.
#    - "out_of_context": Any topic not related to mathematics (e.g., weather, sports, history).

# 2. Parse: If it's a math_problem, extract:
#    - topic: The mathematical category.
#    - variables: List of symbols ($x$, $y$, $P(A)$). For conceptual questions, this may be empty.
#    - constraints: Semantic rules (e.g., "without replacement").

# 3. Validate: 
#    - A numerical problem is 'valid' if it has enough data to solve.
#    - A conceptual question (e.g., "Explain X") is ALWAYS 'valid' as long as the topic is math.

# 4. If NOT math_problem: 
#    - Set topic to "unknown", variables/constraints to [].
#    - Provide a friendly 'direct_response' stating: "I am a specialized Math Mentor. I can only resolve queries related to Math Problems and concepts, especially in the areas of Probability, Algebra, Calculus, and Linear Algebra. How can I help you with math today?"

# GUIDELINES:
# - Preserve LaTeX notation ($x^2$).
# - For probability, look specifically for sampling constraints.
# - CONCEPTUAL RULE: If the user asks for any explanation or definition of a math concept, classify as 'math_problem'.

# CRITICAL:
# - 'topic' MUST be one of: "probability", "algebra", "calculus", "linear_algebra", or "unknown". 
# - Do NOT leave 'topic' as an empty string. If it is not math, use "unknown".
# - 'confidence' must be a JSON number (e.g. 0.95), not a string.
# """

#         try:
#             # Single-pass execution
#             result = self.parser_chain.invoke(prompt)

#             # Update State
#             state["intent"] = result.intent
#             state["intent_confidence"] = result.confidence
#             state["problem_text"] = result.problem_text
#             state["topic"] = result.topic
#             state["variables"] = result.variables
#             state["constraints"] = result.constraints
#             state["parsing_valid"] = result.is_valid
#             state["parsing_issues"] = result.issues
#             state["direct_response"] = result.direct_response
            
#             # Logic for continuing the graph
#             if result.intent != "math_problem" or not result.is_valid:
#                 state["should_continue"] = False
#             else:
#                 state["should_continue"] = True

#             # Add trace for UI requirement
#             state["parser_trace"].append({
#                 "agent": "parser",
#                 "intent_detected": result.intent,
#                 "is_valid": result.is_valid,
#                 "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
#             })

#         except Exception as e:
#             # Handle the specific Groq 400 error if it persists
#             error_msg = str(e)
#             if "expected number, but got string" in error_msg:
#                 state["errors"].append("Groq Type Error: LLM failed to output a raw number. Try re-prompting.")
#             else:
#                 state["errors"].append(f"Parser Error: {error_msg}")
#             state["should_continue"] = False

#         return state

from typing import Dict, Any, List, Literal, Optional
from datetime import datetime
import json
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .state import MathMentorState

# Parser Output Schema 
class ParserOutput(BaseModel):
    """The final structured output from the Parser Agent."""
    intent: Literal["math_problem", "chitchat", "out_of_context"] = Field(
        description="Type of query: math_problem, chitchat, or out_of_context"
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
    )
    topic: Literal["probability", "algebra", "calculus", "linear_algebra", "trigonometry", "geometry", "general_math", "unknown"] = Field(
        default="unknown",
        description="Mathematical topic category"
    )
    problem_text: str = Field(
        default="",
        description="Cleaned version of the input"
    )
    variables: List[str] = Field(
        default_factory=list,
        description="Mathematical variables mentioned"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Constraints or conditions"
    )
    is_valid: bool = Field(
        default=True,
        description="Whether the query can be answered"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Validation issues if any"
    )
    direct_response: Optional[str] = Field(
        default=None,
        description="Direct response for non-math queries"
    )

class ParserAgent:
    """
    Streamlined Parser Agent using structured output with robust error handling.
    """
    def __init__(self, llm: ChatGroq):
        # Use a more reliable approach with JSON mode
        self.llm = llm
        
    def _create_prompt(self, raw_input: str) -> str:
        """Create the parsing prompt."""
        return f"""You are a Math Parser for JEE tutoring. Analyze the user input and return a JSON response.

USER INPUT: "{raw_input}"

CLASSIFICATION RULES:
1. math_problem: ANY math query - solving problems OR explaining concepts
   - Examples: "Solve x²-4=0", "Explain Bayes theorem", "What is derivative?", "Define probability"
   
2. chitchat: Greetings and pleasantries
   - Examples: "hi", "hello", "how are you", "thanks"
   
3. out_of_context: Non-math topics
   - Examples: "weather", "sports", "physics", "history"

RESPONSE FORMAT (valid JSON):
{{
    "intent": "math_problem" | "chitchat" | "out_of_context",
    "confidence": 0.95,
    "topic": "probability" | "algebra" | "calculus" | "linear_algebra" | "unknown",
    "problem_text": "{raw_input}",
    "variables": ["x", "y"],
    "constraints": ["without replacement"],
    "is_valid": true,
    "issues": [],
    "direct_response": null
}}

RULES:
- For math_problem: Set topic, extract variables/constraints, is_valid=true
- For chitchat: topic="unknown", is_valid=false, direct_response="Hello! I'm your JEE Math Mentor..."
- For out_of_context: topic="unknown", is_valid=false, direct_response="I specialize in JEE Math..."
- confidence must be a number 0.0-1.0
- Concept questions (Explain/What is/Define) are math_problem

Return ONLY valid JSON, no other text."""

    def parse(self, state: MathMentorState) -> MathMentorState:
        """
        Parse user input and classify intent.
        
        Args:
            state: Current MathMentorState
            
        Returns:
            Updated state with parsing results
        """
        raw_input = state["raw_input"]
        start_time = datetime.now()

        try:
            # Create prompt
            prompt = self._create_prompt(raw_input)
            
            # Call LLM with JSON mode for more reliable structured output
            response = self.llm.invoke(
                [SystemMessage(content="You are a JSON-only assistant. Return valid JSON."),
                 HumanMessage(content=prompt)],
                temperature=0
            )
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Parse JSON
            result_dict = json.loads(response_text)
            
            # Validate and create ParserOutput with defaults
            result = ParserOutput(
                intent=result_dict.get("intent", "out_of_context"),
                confidence=float(result_dict.get("confidence", 0.9)),
                topic=result_dict.get("topic", "unknown"),
                problem_text=result_dict.get("problem_text", raw_input),
                variables=result_dict.get("variables", []),
                constraints=result_dict.get("constraints", []),
                is_valid=result_dict.get("is_valid", False),
                issues=result_dict.get("issues", []),
                direct_response=result_dict.get("direct_response")
            )

            # Ensure problem_text is set
            if not result.problem_text:
                result.problem_text = raw_input

            # Ensure direct_response is set for non-math
            if result.intent in ["chitchat", "out_of_context"] and not result.direct_response:
                if result.intent == "chitchat":
                    result.direct_response = "Hello! I'm your JEE Math Mentor. I can help you solve problems or explain concepts in Probability, Algebra, Calculus, and Linear Algebra. What would you like to learn today?"
                else:
                    result.direct_response = "I specialize in JEE-level Mathematics. I can help with Probability, Algebra, Calculus, and Linear Algebra. Could you ask a math-related question?"

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
            
            # Logic for continuing
            if result.intent == "math_problem" and result.is_valid:
                state["should_continue"] = True
            else:
                state["should_continue"] = False

            # Add trace
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            state["parser_trace"].append({
                "agent": "parser",
                "intent_detected": result.intent,
                "topic": result.topic,
                "is_valid": result.is_valid,
                "confidence": result.confidence,
                "execution_time_ms": round(execution_time, 2),
                "timestamp": datetime.now().isoformat()
            })

        except json.JSONDecodeError as e:
            # JSON parsing failed
            state["errors"].append(f"Parser Error: Failed to parse LLM response as JSON. {str(e)}")
            self._set_error_defaults(state, raw_input)
            
        except Exception as e:
            # Other errors
            error_msg = str(e)
            
            if "rate limit" in error_msg.lower():
                state["errors"].append("Parser Error: API rate limit reached. Please wait and try again.")
            elif "timeout" in error_msg.lower():
                state["errors"].append("Parser Error: Request timed out. Please try again.")
            else:
                state["errors"].append(f"Parser Error: {error_msg}")
            
            self._set_error_defaults(state, raw_input)

        return state
    
    def _set_error_defaults(self, state: Dict[str, Any], raw_input: str):
        """Set safe defaults when parsing fails."""
        state["intent"] = "out_of_context"
        state["topic"] = "unknown"
        state["problem_text"] = raw_input
        state["variables"] = []
        state["constraints"] = []
        state["parsing_valid"] = False
        state["intent_confidence"] = 0.0
        state["should_continue"] = False
        state["direct_response"] = "I encountered an error processing your request. Please try rephrasing your question."
        
        state["parser_trace"].append({
            "agent": "parser",
            "error": "Failed to parse",
            "timestamp": datetime.now().isoformat()
        })

    def parse_batch(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple inputs in batch (useful for testing).
        
        Args:
            inputs: List of raw input strings
            
        Returns:
            List of parsing results
        """
        results = []
        
        for raw_input in inputs:
            # Create minimal state
            state = {
                "raw_input": raw_input,
                "parser_trace": [],
                "errors": []
            }
            
            # Parse
            parsed_state = self.parse(state)
            
            # Extract results
            results.append({
                "input": raw_input,
                "intent": parsed_state.get("intent"),
                "topic": parsed_state.get("topic"),
                "is_valid": parsed_state.get("parsing_valid"),
                "confidence": parsed_state.get("intent_confidence", 0.0),  # Default to 0.0
                "direct_response": parsed_state.get("direct_response"),
                "errors": parsed_state.get("errors", [])
            })
        
        return results


# Testing utility
# if __name__ == "__main__":
#     from langchain_groq import ChatGroq
#     import os
#     from dotenv import load_dotenv
    
#     load_dotenv()
    
#     model = os.getenv("MODEL")        
#     llm = ChatGroq(model=model, temperature=0)
#     if not llm:
#         print("❌ No model available")
#         exit(1)
    
#     parser = ParserAgent(llm)
    
#     # Test cases
#     test_inputs = [
#         "hi",
#         "hello there",
#         "how are you",
#         "Solve x^2 - 4 = 0",
#         "Explain Bayes' theorem",
#         "What is integration?",
#         "Define probability",
#         "Find the derivative of sin(x)",
#         "Help me with physics",
#         "What's the weather like?",
#         "Calculate 2+2",
#         "A bag has 4 red and 6 black balls. Find probability of drawing 2 red balls.",
#     ]
    
#     print("=" * 80)
#     print("PARSER TESTING")
#     print("=" * 80)
    
#     results = parser.parse_batch(test_inputs)
    
#     for result in results:
#         print(f"\nInput: {result['input']}")
#         print(f"Intent: {result['intent']}")
#         print(f"Topic: {result['topic']}")
#         print(f"Valid: {result['is_valid']}")
#         print(f"Confidence: {result['confidence']}")
#         if result['direct_response']:
#             print(f"Direct Response: {result['direct_response'][:60]}...")
#         if result['errors']:
#             print(f"⚠️  Errors: {result['errors']}")
#         print("-" * 80)
    
#     # Summary
#     print("\n" + "=" * 80)
#     print("SUMMARY")
#     print("=" * 80)
    
#     math_problems = sum(1 for r in results if r['intent'] == 'math_problem')
#     chitchat = sum(1 for r in results if r['intent'] == 'chitchat')
#     out_of_context = sum(1 for r in results if r['intent'] == 'out_of_context')
#     errors = sum(1 for r in results if r['errors'])
    
#     print(f"Math Problems: {math_problems}/{len(results)}")
#     print(f"Chitchat: {chitchat}/{len(results)}")
#     print(f"Out of Context: {out_of_context}/{len(results)}")
#     print(f"Errors: {errors}/{len(results)}")
    
#     # Safe average calculation
#     confidences = [r['confidence'] for r in results if r['confidence'] is not None]
#     if confidences:
#         print(f"Average Confidence: {sum(confidences) / len(confidences):.2f}")
#     else:
#         print("Average Confidence: N/A (all failed)")
    
#     # Expected results
#     print("\n" + "=" * 80)
#     print("EXPECTED vs ACTUAL")
#     print("=" * 80)
#     expected = {
#         "hi": "chitchat",
#         "hello there": "chitchat",
#         "how are you": "chitchat",
#         "Solve x^2 - 4 = 0": "math_problem",
#         "Explain Bayes' theorem": "math_problem",
#         "What is integration?": "math_problem",
#         "Define probability": "math_problem",
#         "Find the derivative of sin(x)": "math_problem",
#         "Help me with physics": "out_of_context",
#         "What's the weather like?": "out_of_context",
#         "Calculate 2+2": "math_problem",
#         "A bag has 4 red and 6 black balls. Find probability of drawing 2 red balls.": "math_problem"
#     }
    
#     correct = 0
#     for result in results:
#         inp = result['input']
#         actual = result['intent']
#         exp = expected.get(inp, "unknown")
#         match = "✅" if actual == exp else "❌"
#         print(f"{match} {inp[:40]:40} | Expected: {exp:15} | Got: {actual}")
#         if actual == exp:
#             correct += 1
    
#     print(f"\nAccuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")