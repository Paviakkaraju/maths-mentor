import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from .hardened_repl import HardenedMathREPL
from .state import MathMentorState


class SolverAgent:
    """
    Agent that solves mathematical problems using sandboxed Python execution.
    Uses a ReAct pattern with enforced tool calling.
    """
    
    def __init__(self, llm, max_iterations: int = 5, verbose: bool = False):
        """
        Initialize the solver agent.
        
        Args:
            llm: Language model (should have temperature=0 for best tool calling)
            max_iterations: Maximum number of ReAct iterations
            verbose: Whether to print debug information
        """
        self.llm = llm
        self.repl = HardenedMathREPL()
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def get_system_prompt(self, state: MathMentorState) -> str:
        """Generate system prompt with strong tool use emphasis."""
        problem = state.get('problem_text', 'No problem specified')
        variables = state.get('variables', [])
        constraints = state.get('constraints', [])
        knowledge = state.get('consolidated_knowledge', '')
        
        var_text = f"\nVariables: {', '.join(variables)}" if variables else ""
        constraint_text = f"\nConstraints: {', '.join(constraints)}" if constraints else ""
        knowledge_text = f"\n\nRelevant Knowledge:\n{knowledge}" if knowledge else ""
        
        return f"""You are a Deterministic Math Solver using Python. 

### PROBLEM CONTEXT
{problem}
{var_text}
{constraint_text}

### CRITICAL FORMATTING RULES (TO PREVENT SYSTEM CRASH)
1. **NO MARKDOWN HEADERS**: Do not use '#' or '##' or '###' anywhere in your response.
2. **NO CODE BLOCKS**: Do not use triple backticks (```) or any markdown code formatting in your text response.
3. **PLAIN TEXT ONLY**: Your conversational thoughts must be simple, plain text sentences.
4. **TOOL CALLS**: All Python code MUST be placed exclusively inside the `python_solver` tool arguments.

### LIBRARIES & ENVIRONMENT
- Libraries are PRE-IMPORTED: `math`, `statistics`, `sympy` (as `sp`), and `numpy` (as `np`).
- **DO NOT** use 'import' statements. Using 'import' will trigger a Security Error.
- Define symbols as: `x = sp.Symbol('x', real=True)`.
- Always assign your final result to a variable named `result`.

### WORKFLOW
1. **PLAN**: State your strategy in one plain-text sentence. 
2. **ACTION**: Call the `python_solver` tool immediately.
3. **OBSERVE**: Use the tool's output to form your next thought or final answer.

### EXAMPLE OF CORRECT BEHAVIOR
Plan: I will calculate the combinations for the total and favorable outcomes.
[CALL python_solver with code="nS = math.comb(10, 2); nE = math.comb(4, 2); result = nE/nS"]

### EXAMPLE OF WRONG BEHAVIOR (CRASHES THE SYSTEM)
"## Plan: I will use math. 
Code: ```python 
import math... 
```" 
-> NEVER DO THIS. Headers and backticks cause a 400 Bad Request error.

{knowledge_text}

Remember: Be concise, be plain-text, and CALL THE TOOL."""

    def solve(self, state: MathMentorState) -> Dict[str, Any]:
        """
        Main solving logic with enforced tool calling.
        
        Args:
            state: Current state containing problem and context
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = datetime.now()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting solver for: {state.get('problem_text', 'Unknown')[:100]}")
            print(f"{'='*60}\n")
        
        # Define the tool
        @tool
        def python_solver(code: str) -> str:
            """Execute safe Python code for mathematical computations.
            
            CRITICAL: This tool EXECUTES code. You must CALL this tool.
            
            PRE-IMPORTED: math, statistics, sympy (sp), numpy (np)
            
            Args:
                code: Python code to execute (no import statements)
                
            Returns:
                Execution result or error message
            """
            if self.verbose:
                print(f"[TOOL CALLED] Executing code:\n{code}\n")
            
            result = self.repl.execute(code)
            
            if self.verbose:
                print(f"[TOOL RESULT] {result}\n")
            
            return result
        
        # Bind tool to LLM
        llm_with_tools = self.llm.bind_tools([python_solver])
        
        # Initialize conversation with STRONG instruction to use tool
        messages = [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content="""**CRITICAL INSTRUCTION**

                You MUST use the python_solver TOOL to solve this problem.

                DO NOT write code in your response text.
                DO NOT explain code without calling the tool.
                ACTUALLY CALL the python_solver tool with your code.

                Process:
                1. Think: What's my approach? (state your PLAN)
                2. Action: CALL python_solver tool with code
                3. Observe: See the result
                4. If done: State final answer
                5. If not done: CALL python_solver again

                Start by CALLING the python_solver tool now.""")
        ]
        
        # Tracking variables
        structured_trace = []
        final_answer = None
        iteration_count = 0
        error_occurred = False
        tool_was_called = False
        
        # Manual ReAct Loop
        for iteration in range(self.max_iterations):
            iteration_count += 1
            
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            try:
                # Get LLM response
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                # Check if tool was called
                if not response.tool_calls:
                    if self.verbose:
                        print(f"[WARNING] No tool calls in iteration {iteration + 1}")
                        print(f"Response: {response.content[:200]}...")
                    
                    # First iteration without tool call = push harder
                    if iteration == 0 and not tool_was_called:
                        if self.verbose:
                            print("[ACTION] Prompting LLM to use tool...")
                        
                        messages.append(HumanMessage(
                            content="""You did NOT call the python_solver tool. 

                                    I can see you wrote code or explanation in text, but you must CALL THE TOOL.

                                    Click/invoke the python_solver tool and put your code in the arguments.

                                    CALL THE TOOL NOW - don't write code in text!"""
                        ))
                        continue
                    else:
                        # LLM provided final answer after using tool
                        final_answer = response.content
                        
                        if self.verbose:
                            print(f"[INFO] Final answer received")
                        
                        break
                
                # Mark that tool was called
                tool_was_called = True
                
                # Process tool calls
                for tool_call in response.tool_calls:
                    try:
                        # Extract plan from response
                        plan = self._extract_plan(response.content) if response.content else "Executing calculation"
                        
                        # Get code from tool call
                        code = tool_call["args"].get("code", "")
                        
                        if not code:
                            error_msg = "Tool called but no code provided"
                            if self.verbose:
                                print(f"[ERROR] {error_msg}")
                            
                            messages.append(
                                ToolMessage(
                                    content=f"Error: {error_msg}",
                                    tool_call_id=tool_call["id"]
                                )
                            )
                            continue
                        
                        # Add to trace
                        trace_entry = {
                            "iteration": iteration + 1,
                            "plan": plan,
                            "code": code,
                            "timestamp": datetime.now().isoformat()
                        }
                        structured_trace.append(trace_entry)
                        
                        # Execute tool
                        result = python_solver.invoke(tool_call["args"])
                        
                        # Validate result
                        validation = self._validate_result(result)
                        
                        # Add result to trace
                        result_entry = {
                            "iteration": iteration + 1,
                            "result": result,
                            "is_error": validation["is_error"],
                            "is_timeout": validation["is_timeout"],
                            "is_security_error": validation["is_security_error"],
                            "timestamp": datetime.now().isoformat()
                        }
                        structured_trace.append(result_entry)
                        
                        # Check for critical errors
                        if validation["is_security_error"] or validation["is_timeout"]:
                            error_occurred = True
                        
                        # Feed result back to LLM
                        messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call["id"]
                            )
                        )
                        
                    except Exception as e:
                        error_msg = f"Error processing tool call: {str(e)}"
                        
                        if self.verbose:
                            print(f"[ERROR] {error_msg}")
                        
                        structured_trace.append({
                            "iteration": iteration + 1,
                            "error": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call["id"]
                            )
                        )
                        error_occurred = True
                
            except Exception as e:
                error_msg = f"Error in iteration {iteration + 1}: {str(e)}"
                
                if self.verbose:
                    print(f"[ERROR] {error_msg}")
                
                structured_trace.append({
                    "iteration": iteration + 1,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                error_occurred = True
                break
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Handle edge cases
        if not tool_was_called:
            final_answer = "ERROR: LLM did not call the python_solver tool despite multiple prompts."
            error_occurred = True
            
            if self.verbose:
                print(f"[CRITICAL] Tool was never called!")
        
        if final_answer is None and iteration_count >= self.max_iterations:
            final_answer = f"Maximum iterations ({self.max_iterations}) reached. Check trace for last result."
            
            if self.verbose:
                print(f"[WARNING] Max iterations reached")


        # Extract specific fields for the global state
        solution_steps = [t['plan'] for t in structured_trace if 'plan' in t]
        calculations = [
            {"code": t['code'], "result": t.get('result')} 
            for t in structured_trace if 'code' in t
        ]

        # Prepare result
        result = {
            "final_answer": final_answer or "No answer generated",
            "solution_steps": solution_steps,              # Mapping to state
            "calculations_performed": calculations,
            "solver_trace": structured_trace,
            "execution_time_ms": round(execution_time, 2),
            "iteration_count": iteration_count,
            # "success": not error_occurred and final_answer is not None and tool_was_called,
            "success": (final_answer is not None and final_answer != "" and not state.get("errors_unrecovered", False) and
                        state.get("parsing_valid", True) and tool_was_called),
            "error_occurred": error_occurred,
            "tool_was_called": tool_was_called,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Solver completed in {execution_time:.2f}ms")
            print(f"Iterations: {iteration_count}/{self.max_iterations}")
            print(f"Tool called: {tool_was_called}")
            print(f"Success: {result['success']}")
            print(f"{'='*60}\n")
        
        return result
    
    def _extract_plan(self, content: str) -> str:
        """Extract plan from LLM response."""
        if not content:
            return "Executing calculation"
        
        # Look for PLAN: marker
        if "PLAN:" in content.upper():
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'PLAN:' in line.upper():
                    plan_text = line.split(':', 1)[1].strip()
                    # Include next line if it's part of the plan
                    if i + 1 < len(lines) and lines[i + 1].strip() and not any(
                        marker in lines[i + 1].upper() for marker in ['CODE:', 'RESULT:', '```', 'CALL']
                    ):
                        plan_text += " " + lines[i + 1].strip()
                    return plan_text
        
        # Fallback: first meaningful line
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('```'):
                return line[:200]
        
        return "Solving problem"
    
    def _validate_result(self, result: str) -> Dict[str, Any]:
        """Validate execution result."""
        is_error = any(
            error_type in result 
            for error_type in ['Error:', 'Exception:', 'Traceback']
        )
        
        return {
            "is_error": is_error,
            "is_timeout": "Timeout" in result,
            "is_security_error": "Security Error" in result,
            "result": result
        }
    
    def solve_batch(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Solve multiple problems in batch."""
        results = []
        
        for i, problem in enumerate(problems):
            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"Problem {i+1}/{len(problems)}")
                print(f"{'#'*60}")
            
            result = self.solve(problem)
            result["problem_index"] = i
            results.append(result)
        
        return results
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics across multiple solve results."""
        if not results:
            return {}
        
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        tool_called = sum(1 for r in results if r.get("tool_was_called", False))
        errors = sum(1 for r in results if r.get("error_occurred", False))
        
        execution_times = [r.get("execution_time_ms", 0) for r in results]
        iterations = [r.get("iteration_count", 0) for r in results]
        
        return {
            "total_problems": total,
            "successful": successful,
            "failed": total - successful,
            "tool_called_count": tool_called,
            "tool_call_rate": round(tool_called / total * 100, 2) if total > 0 else 0,
            "error_occurred": errors,
            "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
            "avg_execution_time_ms": round(sum(execution_times) / total, 2) if total > 0 else 0,
            "max_execution_time_ms": max(execution_times) if execution_times else 0,
            "min_execution_time_ms": min(execution_times) if execution_times else 0,
            "avg_iterations": round(sum(iterations) / total, 2) if total > 0 else 0,
            "max_iterations_used": max(iterations) if iterations else 0
        }