from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from .state import MathMentorState, create_initial_state
from .parser_agent import ParserAgent
from .intent_router import IntentRouterAgent
from .rag_agent import RAGAgent
from .solver_agent import SolverAgent
from .verifier_agent import VerifierAgent
from .explainer_agent import ExplainerAgent

class MathMentorGraph:
    def __init__(self, base_llm_name: str, chroma_path: str):
        
        if base_llm_name is None:
            raise ValueError("base_llm_name cannot be None. Check your runner.py or .env file.")
            
        print(f"Initializing Graph with model: {base_llm_name}")

        # 1. Create specialized LLM instances
        # Logic LLM: Cold and precise
        self.logic_llm = ChatGroq(model=base_llm_name, temperature=0.01)
        
        # Creative LLM: Warm and engaging
        self.creative_llm = ChatGroq(model=base_llm_name, temperature=0.6)

        # 2. Assign them to the agents based on their role
        self.parser = ParserAgent(self.logic_llm)
        self.router = IntentRouterAgent(self.logic_llm)
        self.rag = RAGAgent(self.logic_llm, chroma_path=chroma_path)
        self.solver = SolverAgent(self.logic_llm)
        self.verifier = VerifierAgent(self.logic_llm)
        
        # The Explainer gets the 'warm' model
        self.explainer = ExplainerAgent(self.creative_llm)
        
        self.workflow = self._create_graph()
        
        # Build the Graph
        self.workflow = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(MathMentorState)

        # 1. ADD NODES
        workflow.add_node("parser", self.parser.parse)
        workflow.add_node("router", self.router.route)
        workflow.add_node("rag", self.rag.retrieve_and_consolidate)
        workflow.add_node("solver", self.solver.solve)
        workflow.add_node("verifier", self.verifier.verify)
        workflow.add_node("explainer", self.explainer.explain)

        # 2. DEFINE EDGES & ROUTING LOGIC

        # Entry Point
        workflow.set_entry_point("parser")

        # GATEKEEPER: From Parser, decide if we continue to Math Logic
        def gatekeeper_router(state: MathMentorState) -> Literal["router", "__end__"]:
            if state.get("intent") == "math_problem" and state.get("should_continue"):
                return "router"
            return "__end__"

        workflow.add_conditional_edges("parser", gatekeeper_router)

        # Router always goes to RAG to get context
        workflow.add_edge("router", "rag")

        # STRATEGY ROUTER: From RAG, decide if we solve or just explain
        def strategy_router(state: MathMentorState) -> Literal["solver", "explainer"]:
            if state.get("workflow_type") == "conceptual":
                return "explainer"
            return "solver"

        workflow.add_conditional_edges("rag", strategy_router)

        # Solver always goes to Verifier
        workflow.add_edge("solver", "verifier")

        # RELIABILITY LOOP: From Verifier, check if we need to retry or finish
        def reliability_router(state: MathMentorState) -> Literal["explainer", "solver", "__end__"]:
            if state.get("is_correct"):
                return "explainer"
            
            # Retry logic
            if state.get("solver_trials", 0) < 1:
                return "solver"
            
            # If still wrong after retry, we end (or go to HITL)
            return "explainer" # Still go to explainer to show the best effort/error

        workflow.add_conditional_edges("verifier", reliability_router)

        # Explainer is the final stop
        workflow.add_edge("explainer", END)

        return workflow.compile()

    def run(self, input_text: str, session_id: str):
        """Execute the graph for a given user input."""
    
        initial_state = create_initial_state(input_text, session_id)
        return self.workflow.invoke(initial_state)