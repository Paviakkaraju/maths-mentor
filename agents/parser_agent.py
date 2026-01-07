"""
Parser Agent - Intent Classification + Parsing with ReAct Pattern
This agent:
1. Classifies intent (math/chitchat/OOC) using LLM
2. Parses math problems into structured format using LLM
3. Responds appropriately to non-math inputs
4. Uses tools autonomously in a Think-Act-Observe loop
"""

from typing import Dict, Any, List, Literal, Optional, Annotated
from datetime import datetime
import json
import operator

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

from .state import MathMentorState


# LLM Output Schemas

class IntentResult(BaseModel):
    """Schema for intent classification result."""
    intent: str = Field(
        description="One of: math_problem, chitchat, out_of_context"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Short explanation of why this intent was chosen"
    )


class ParsedMathProblem(BaseModel):
    """Schema for parsed math problem."""
    problem_text: str = Field(description="Original problem text, cleaned.")
    topic: Literal["probability", "algebra", "calculus", "linear_algebra", "unknown"] = Field(
        description="Main topic of the math problem."
    )
    variables: List[str] = Field(
        description="List of variable symbols explicitly used in the problem."
    )
    constraints: List[str] = Field(
        description="Natural-language constraints or conditions extracted from the problem."
    )
    # reasoning: Optional[str] = Field(
    #     default=None,
    #     description="Short explanation of how you identified topic, variables, and constraints."
    # )


# Agent State for Internal Loop

class ParserAgentState(MessagesState):
    """Internal state for parser agent's ReAct loop."""
    # Inherits 'messages' from MessagesState
    intent_result: Optional[Dict[str, Any]] = None
    parse_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    should_continue_loop: bool = True


# Tools

def make_parser_tools(base_llm: ChatGroq):
    """Create all parser tools with LLM backing."""
    
    # Intent classifier tool
    intent_llm = base_llm.with_structured_output(IntentResult)
    
    @tool
    def classify_intent(text: str) -> str:
        """
        Classify user's message intent as math_problem, chitchat, or out_of_context.
        
        Args:
            text: The user's message to classify
            
        Returns:
            JSON string with intent, confidence, and reasoning
        """
        result = intent_llm.invoke(
            f"""You are classifying a user's message for a math tutoring system.

            User message: {text}

            Classify the intent as:
            - "math_problem" if the user is asking a math question or giving a math problem.
            - "chitchat" if the user is greeting, thanking, or casually talking without asking for math help.
            - "out_of_context" if the user is asking something not about math tutoring.
            """
        )
        return json.dumps(result.model_dump())
    
    # Math parser tool
    parser_llm = base_llm.with_structured_output(ParsedMathProblem)
    
    @tool
    def parse_math_problem(text: str) -> str:
        """
        Parse a math problem into structured components.
        
        Args:
            text: The math problem text to parse
            
        Returns:
            JSON string with problem_text, topic, variables, constraints
        """
        result = parser_llm.invoke(
            f"""Extract structured information from this math problem.

                Problem: "{text}"

                Extract EXACTLY:
                - problem_text: cleaned problem text (string)
                - topic: ONE OF ["probability", "algebra", "calculus", "linear_algebra", "unknown"]
                - variables: list of variable symbols like ["x", "y"] (can be empty list)
                - constraints: list of constraint strings (can be empty list)

                Return ONLY the extraction structure, no markdown, no explanations."""
        )
        return json.dumps(result.model_dump())
    
    @tool
    def validate_parsing(parsed_json: str) -> str:
        """
        Validate if a parsed math problem is complete and valid.
        
        Args:
            parsed_json: JSON string of the parsed problem
            
        Returns:
            JSON string with is_valid (bool) and issues (list)
        """
        try:
            parsed = json.loads(parsed_json)
            issues = []
            
            if not parsed.get("problem_text"):
                issues.append("Problem text is empty")
            if parsed.get("topic") == "unknown":
                issues.append("Could not identify topic - may need clarification")
            if len(parsed.get("problem_text", "")) < 10:
                issues.append("Problem text too short - may be incomplete")
            
            return json.dumps({
                "is_valid": len(issues) == 0,
                "issues": issues
            })
        except Exception as e:
            return json.dumps({
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"]
            })
    
    return [classify_intent, parse_math_problem, validate_parsing]


# Parser Agent

class ParserAgent:
    """
    Parser Agent with ReAct pattern using current LangGraph API.
    
    Uses StateGraph to build a custom ReAct loop:
    - llm_node: Agent thinks and decides which tools to call
    - tool_node: Executes the selected tools
    - Loop continues until agent decides it's done
    """
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.tools = make_parser_tools(llm)
        
        # Bind tools to LLM
        self.llm_with_tools = llm.bind_tools(self.tools)
        
        # Build the ReAct graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the ReAct workflow graph."""
        
        # Create workflow
        workflow = StateGraph(ParserAgentState)
        
        # Add nodes
        workflow.add_node("llm", self._llm_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Set entry point
        workflow.set_entry_point("llm")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "llm",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Tools always go back to LLM
        workflow.add_edge("tools", "llm")
        
        return workflow.compile()
    
    def _llm_node(self, state: ParserAgentState) -> ParserAgentState:
        """LLM reasoning node - thinks and potentially calls tools."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        
        # Add response to messages
        return {"messages": [response]}
    
    def _should_continue(self, state: ParserAgentState) -> str:
        """Decide whether to continue tool calling or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Otherwise, we're done
        return "end"
    
    def parse(self, state: MathMentorState) -> MathMentorState:
        """
        Main entry point - runs the ReAct loop and updates MathMentorState.
        
        Args:
            state: Current MathMentorState
            
        Returns:
            Updated MathMentorState
        """
        raw_input = state["raw_input"]
        
        # Create initial messages for the agent
        initial_messages = [
            HumanMessage(content=f"""You are a Parser Agent for a math tutoring system.

                USER INPUT: "{raw_input}"

                YOUR WORKFLOW:

                STEP 1: Call classify_intent tool to determine intent.

                STEP 2: Based on the result:

                A) If intent is "chitchat":
                - After seeing the tool result, respond warmly: "Hi! I'm a math tutor. I can help with probability, algebra, calculus, and linear algebra. What math problem are you working on?"
                - DO NOT call any more tools
                - Just provide this response and stop

                B) If intent is "out_of_context":
                - After seeing the tool result, respond politely: "I'm specialized in math tutoring. I can't help with that, but I'd love to help you with a math problem!"
                - DO NOT call any more tools
                - Just provide this response and stop

                C) If intent is "math_problem":
                - Call parse_math_problem tool
                - Then call validate_parsing tool
                - Then stop

                CRITICAL:
                - For chitchat and out_of_context, provide your response IMMEDIATELY after classify_intent
                - DO NOT call any other tools for chitchat/OOC
                - DO NOT say "I'll respond now" or explain what you're doing
                - Just give the appropriate response

                Start by calling classify_intent.""")
        ]
        
        try:
            # Run the ReAct graph
            agent_state = {"messages": initial_messages}
            result = self.graph.invoke(agent_state)
            
            # Extract results
            messages = result["messages"]
            final_message = messages[-1]
            
            # Extract trace
            trace = self._extract_trace(messages)
            state["parser_trace"].extend(trace)
            
            # Parse tool results from messages
            intent_result = self._find_tool_result(messages, "classify_intent")
            if intent_result:
                state["intent"] = intent_result.get("intent")
                state["intent_confidence"] = float(intent_result.get("confidence", 0.0))
            
            # Handle different intents
            if state["intent"] in ["chitchat", "out_of_context"]:
                # Get final response from agent
                if hasattr(final_message, "content"):
                    state["direct_response"] = final_message.content
                state["should_continue"] = False
            
            elif state["intent"] == "math_problem":
                # Extract parsing results
                parse_result = self._find_tool_result(messages, "parse_math_problem")
                if parse_result:
                    state["problem_text"] = parse_result.get("problem_text", raw_input)
                    state["topic"] = parse_result.get("topic", "unknown")
                    state["variables"] = parse_result.get("variables", [])
                    state["constraints"] = parse_result.get("constraints", [])
                
                # Extract validation results
                validation = self._find_tool_result(messages, "validate_parsing")
                if validation:
                    state["parsing_valid"] = bool(validation.get("is_valid", False))
                    state["parsing_issues"] = validation.get("issues", [])
                
                state["should_continue"] = state["parsing_valid"]
            
            else:
                # Unknown intent - treat as out of context
                state["intent"] = "out_of_context"
                state["intent_confidence"] = 0.0
                state["direct_response"] = (
                    "I'm specialized in math tutoring (probability, algebra, "
                    "calculus, and linear algebra). I can't help with that, "
                    "but I'd love to help you with a math problem!"
                )
                state["should_continue"] = False
        
        except Exception as e:
            error_msg = f"Parser error: {str(e)}"
            state["errors"].append(error_msg)
            state["should_continue"] = False
            state["parser_trace"].append({
                "type": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _extract_trace(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract ReAct trace from messages."""
        trace = []
        
        for msg in messages:
            # Check for tool calls (agent deciding to use tools)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    trace.append({
                        "type": "tool_call",
                        "tool": tool_call.get("name", "unknown"),
                        "args": tool_call.get("args", {}),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Check for agent thoughts
            if hasattr(msg, "content") and msg.content:
                content_lower = str(msg.content).lower()
                if any(word in content_lower for word in ["think", "reason", "because", "first", "next"]):
                    trace.append({
                        "type": "thought",
                        "content": str(msg.content)[:200],
                        "timestamp": datetime.now().isoformat()
                    })
        
        return trace
    
    def _find_tool_result(self, messages: List[Any], tool_name: str) -> Dict[str, Any]:
        """Find the result of a specific tool call in messages."""
        for i, msg in enumerate(messages):
            # Look for tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == tool_name:
                        # Look for the corresponding tool result
                        # Tool results come as ToolMessage after the tool call
                        for j in range(i + 1, len(messages)):
                            next_msg = messages[j]
                            if isinstance(next_msg, ToolMessage):
                                try:
                                    # Tool returns JSON string
                                    return json.loads(next_msg.content)
                                except:
                                    pass
        
        return {}


# ============ EXAMPLE USAGE ============

# if __name__ == "__main__":
#     from langchain_groq import ChatGroq
#     from state import create_initial_state
#     import os
    
#     # Check API key
#     if "GROQ_API_KEY" not in os.environ:
#         print("❌ Please set GROQ_API_KEY")
#         exit(1)
    
#     # Initialize
#     llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
#     parser = ParserAgent(llm)
    
#     # Test 1: Math problem
#     print("="*60)
#     print("TEST 1: Math Problem")
#     print("="*60)
#     state = create_initial_state(
#         "What is the probability of rolling a sum of 7 with two dice?",
#         "test_001"
#     )
#     result = parser.parse(state)
#     print(f"Intent: {result['intent']}")
#     print(f"Confidence: {result['intent_confidence']:.2f}")
#     print(f"Should continue: {result['should_continue']}")
#     if result['should_continue']:
#         print(f"Problem: {result['problem_text']}")
#         print(f"Topic: {result['topic']}")
#         print(f"Variables: {result['variables']}")
    
#     # Test 2: Chitchat
#     print("\n" + "="*60)
#     print("TEST 2: Chitchat")
#     print("="*60)
#     state2 = create_initial_state("Hey, how are you?", "test_002")
#     result2 = parser.parse(state2)
#     print(f"Intent: {result2['intent']}")
#     print(f"Should continue: {result2['should_continue']}")
#     print(f"Response: {result2['direct_response']}")
    
#     # Test 3: Out of context
#     print("\n" + "="*60)
#     print("TEST 3: Out of Context")
#     print("="*60)
#     state3 = create_initial_state("What's the weather like?", "test_003")
#     result3 = parser.parse(state3)
#     print(f"Intent: {result3['intent']}")
#     print(f"Should continue: {result3['should_continue']}")
#     print(f"Response: {result3['direct_response']}")
    
#     # View trace
#     print("\n" + "="*60)
#     print("AGENT TRACE (Test 1)")
#     print("="*60)
#     for i, trace_item in enumerate(result['parser_trace'], 1):
#         print(f"{i}. [{trace_item['type']}]", end=" ")
#         if trace_item['type'] == 'tool_call':
#             print(f"{trace_item['tool']}")
#         elif trace_item['type'] == 'thought':
#             print(f"{trace_item['content'][:60]}...")