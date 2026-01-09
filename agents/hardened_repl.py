import ast
import math
import statistics
import sympy
import numpy as np
import threading
import io
import sys
import time
import queue
from typing import Any, Dict, Optional


class SecurityError(Exception):
    """Raised when code violates security constraints"""
    pass


class TimeoutError(Exception):
    """Raised when execution exceeds timeout"""
    pass


class HardenedMathREPL:
    """
    A strictly sandboxed Python executor for mathematical computations.
    Implements AST node filtering, attribute access control, and timeout protection.
    
    Note: Uses threading for timeout (works on Windows). This means Python-level
    infinite loops can still consume CPU, but execution will be terminated after timeout.
    For production, consider using Docker containers with resource limits.
    """
    
    def __init__(self, timeout: int = 5, max_output_len: int = 1000, max_code_len: int = 10000):
        self.timeout = timeout
        self.max_output_len = max_output_len
        self.max_code_len = max_code_len
        
        # Allowed libraries
        self.allowed_libs = {
            "math": math,
            "statistics": statistics,
            "sympy": sympy,
            "np": np,
            "numpy": np,
            "sp": sympy
        }
        
        # Explicitly allowed AST nodes
        self.safe_nodes = {
            # Core
            'Module', 'Expr', 'Assign', 'AugAssign', 'Name', 'Load', 'Store',
            
            # Literals
            'Num', 'Str', 'Constant', 'List', 'Tuple', 'Dict',
            
            # Operations
            'BinOp', 'UnaryOp', 'Compare', 'BoolOp',
            
            # Operators
            'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow', 
            'MatMult', 'USub', 'UAdd',
            
            # Boolean operators
            'And', 'Or', 'Not',
            
            # Comparisons
            'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn',
            
            # Function calls
            'Call', 'arg', 'keyword', 'arguments',
            
            # Subscripting (for arrays)
            'Subscript', 'Index', 'Slice',
            
            # Attribute access (needed for module.function calls)
            'Attribute',
            
            # Conditionals (if/else) - useful for math
            'If', 'IfExp',
            
            # Comprehensions (safe for mathematical operations)
            'ListComp', 'DictComp', 'SetComp',
            'comprehension', 'Set',
        }
        
        # Safe attributes that can be accessed (whitelist)
        self.safe_attributes = {
            # SymPy functions
            'Symbol', 'symbols', 'solve', 'simplify', 'expand', 'factor', 
            'diff', 'integrate', 'limit', 'series', 'apart', 'together',
            'cancel', 'nsimplify', 'lambdify', 'evalf', 'subs',
            'Matrix', 'det', 'inv', 'eigenvals', 'eigenvects',
            'latex', 'pretty', 'pprint',
            
            # Math functions (both math and sympy versions)
            'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
            'log', 'log10', 'log2', 'exp', 'pow',
            'ceil', 'floor', 'trunc', 'fabs', 'factorial',
            'gcd', 'lcm', 'degrees', 'radians', 'copysign',
            'isfinite', 'isinf', 'isnan',
            
            # NumPy functions and methods
            'array', 'mean', 'median', 'std', 'var', 'min', 'max', 'sum',
            'prod', 'cumsum', 'cumprod', 'sort', 'argsort',
            'linalg', 'dot', 'matmul', 'cross', 'inner', 'outer',
            'transpose', 'reshape', 'flatten', 'ravel',
            'linspace', 'arange', 'logspace', 'zeros', 'ones', 'eye', 'diag',
            'concatenate', 'vstack', 'hstack', 'split',
            'clip', 'round', 'abs',
            
            # Statistics functions
            'stdev', 'variance', 'mode', 'quantiles', 'correlation',
            'median_low', 'median_high', 'median_grouped',
            
             # PROBABILITY & COMBINATORICS
            'comb', 'perm', 'gamma', 'erf', 'erfc',
            
            # CRITICAL SYMPY CLASSES & LOGIC
            'Eq', 'Rational', 'Integer', 'Float', 'Poly', 'solveset',
            'Derivative', 'Integral', 'Limit', 'Function', 'dsolve',
            'root', 'real_root', 'Abs',
            
            # ADDITIONAL TRIGONOMETRY
            'sec', 'csc', 'cot', 'asec', 'acsc', 'acot',
            
            # LINEAR ALGEBRA ADDITIONS
            'trace', 'norm', 'eig', 'eigvals', 'eigvecs',
            
            # SET THEORY (SymPy)
            'Union', 'Intersection', 'Complement', 'FiniteSet', 'Interval',

            # Object properties
            'real', 'imag',  # Complex numbers
            'shape', 'dtype', 'T', 'ndim', 'size',  # NumPy arrays
            
            # Math/SymPy constants
            'pi', 'e', 'inf', 'nan', 'tau',  # math module
            'I', 'E', 'Pi', 'oo', 'zoo', 'nan',  # SymPy
            
            # String methods (for formatting results)
            'format', 'join', 'split', 'strip', 'upper', 'lower',
            'replace', 'startswith', 'endswith',
            
            # List/array methods
            'append', 'extend', 'insert', 'remove', 'pop', 'index', 'count',
            'reverse', 'copy',
        }
        
        # Safe builtins to expose
        self.safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "round": round,
            "pow": pow,
            "sorted": sorted,
            "reversed": reversed,
            "all": all,
            "any": any,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "type": type,
            "print": print,  # Allow print for debugging
        }
    
    def _check_ast(self, code: str) -> None:
        """
        Validate AST to ensure only safe operations are used.
        
        Raises:
            SecurityError: If forbidden constructs are detected
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax: {e}")
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            
            # Check if node type is allowed
            if node_type not in self.safe_nodes:
                raise SecurityError(
                    f"Forbidden AST node: {node_type}. "
                    f"Only mathematical operations are allowed."
                )
            
            # Block all imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityError(
                    "Import statements are not allowed. "
                    "Use pre-imported libraries: math, statistics, sympy, numpy"
                )
            
            # Block function and class definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, 
                                ast.ClassDef, ast.Lambda)):
                raise SecurityError(
                    "Function and class definitions are not allowed. "
                    "Use only expressions and assignments."
                )
            
            # Block loops (to prevent infinite loops)
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                raise SecurityError(
                    "Loops are not allowed to prevent infinite loops. "
                    "Use NumPy vectorized operations or list comprehensions instead."
                )
            
            # Block attribute access except whitelisted
            if isinstance(node, ast.Attribute):
                attr_name = node.attr
                
                # Block all private/magic attributes
                if attr_name.startswith('_'):
                    raise SecurityError(
                        f"Access to private/magic attributes is forbidden: {attr_name}"
                    )
                
                # Check whitelist
                if attr_name not in self.safe_attributes:
                    raise SecurityError(
                        f"Attribute access not allowed: {attr_name}. "
                        f"If this is a legitimate mathematical operation, please contact support."
                    )
            
            # Limit call complexity
            if isinstance(node, ast.Call):
                call_depth = len(list(ast.walk(node)))
                if call_depth > 50:
                    raise SecurityError(
                        f"Expression too complex (depth: {call_depth}). "
                        "Break it into simpler steps."
                    )
    
    def _execute_with_timeout(self, code: str, safe_globals: dict, local_vars: dict) -> str:
        """
        Execute code in a thread with timeout protection.
        
        Args:
            code: Python code to execute
            safe_globals: Global namespace
            local_vars: Local namespace
            
        Returns:
            Execution result as string
            
        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails
        """
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                # Execute code
                exec(code, safe_globals, local_vars)
                
                # Restore stdout
                sys.stdout = old_stdout
                output = captured_output.getvalue()
                
                # Determine result
                if 'result' in local_vars:
                    result_str = str(local_vars['result'])
                elif output.strip():
                    result_str = output.strip()
                else:
                    # Get last assigned variable
                    if local_vars:
                        last_var = list(local_vars.keys())[-1]
                        result_str = f"{last_var} = {local_vars[last_var]}"
                    else:
                        result_str = "Execution successful (no output)"
                
                result_queue.put(result_str)
                
            except Exception as e:
                exception_queue.put(e)
        
        # Start thread
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)
        
        # Check if timed out
        if thread.is_alive():
            # Thread is still running - it timed out
            # Note: We can't actually kill the thread in Python, but we can stop waiting
            raise TimeoutError(
                f"Execution exceeded {self.timeout} seconds. "
                "Simplify your calculation or break it into steps. "
                "(Note: The thread may still be running in the background)"
            )
        
        # Check for exceptions
        if not exception_queue.empty():
            exc = exception_queue.get()
            raise RuntimeError(f"{type(exc).__name__}: {str(exc)}")
        
        # Get result
        if result_queue.empty():
            raise RuntimeError("Execution completed but produced no result")
        
        return result_queue.get()
    
    def execute(self, code: str) -> str:
        """
        Execute code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result or error message
        """
        # Input validation
        if not code or not code.strip():
            return "Error: Empty code provided"
        
        if len(code) > self.max_code_len:
            return f"Error: Code too long (max {self.max_code_len} characters)"
        
        try:
            # Validate AST
            self._check_ast(code)
            
            # Create isolated namespace
            safe_globals = {
                "__builtins__": self.safe_builtins,
                **self.allowed_libs
            }
            
            local_vars = {}
            
            # Execute with timeout
            result = self._execute_with_timeout(code, safe_globals, local_vars)
            
            # Truncate output if needed
            if len(result) > self.max_output_len:
                result = result[:self.max_output_len] + "\n... (output truncated)"
            
            return result
            
        except SecurityError as e:
            return f"Security Error: {str(e)}"
        except SyntaxError as e:
            return f"Syntax Error: {str(e)}"
        except TimeoutError as e:
            return f"Timeout Error: {str(e)}"
        except Exception as e:
            return f"Execution Error: {type(e).__name__}: {str(e)}"


# Example usage and tests
if __name__ == "__main__":
    repl = HardenedMathREPL()
    
    # Test cases - libraries are pre-imported, no need for import statements
    test_cases = [
        # Valid math with sympy (pre-imported)
        ("SymPy equation solving", """
x = sympy.Symbol('x', real=True)
result = sympy.solve(x**2 - 4, x)
"""),
        # Valid numpy (pre-imported as np)
        ("NumPy statistics", """
arr = np.array([1, 2, 3, 4, 5])
result = np.mean(arr)
"""),
        # Valid math operations
        ("Math functions", """
result = math.sqrt(16) + math.sin(math.pi/2)
"""),
        # Should fail - private attribute access
        ("Private attribute (should fail)", """
result = math.__dict__
"""),
        # Should fail - trying to import (not needed, but blocked anyway)
        ("Import statement (should fail)", """
import os
result = os.listdir()
"""),
        # Should fail - class definition
        ("Class definition (should fail)", """
class MyClass:
    pass
result = MyClass()
"""),
        # Should fail - loop
        ("Loop (should fail)", """
total = 0
for i in range(10):
    total += i
result = total
"""),
        # Valid - list comprehension instead of loop
        ("List comprehension (should work)", """
result = sum([i for i in range(10)])
"""),
        # Valid - complex sympy operation
        ("Complex SymPy", """
x = sympy.Symbol('x', real=True)
expr = x**3 - 3*x**2 + 2*x
result = sympy.factor(expr)
"""),
    ]
    
    for name, code in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"Code: {code.strip()[:100]}...")
        print(f"{'-'*60}")
        result = repl.execute(code)
        print(f"Result: {result}")