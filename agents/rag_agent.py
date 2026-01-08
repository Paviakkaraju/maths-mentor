import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_groq import ChatGroq
import chromadb
from .state import MathMentorState
from rag.embeddings import get_embeddings

class RAGAgent:
    """
    Specialist Agent for Knowledge Retrieval and Consolidation.
    Provides the Solver and Explainer with grounded math facts.
    """
    def __init__(self, llm: ChatGroq, chroma_path: str = "chromadb"):
        self.llm = llm
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection("maths_docs")
        self.embeddings = get_embeddings()

    def retrieve_and_consolidate(self, state: MathMentorState) -> Dict[str, Any]:
        """
        The main entry point for the RAG node.
        """
        topic = state.get("topic", "unknown").lower()
        db_value = f"{topic}.txt"
        query = state.get("problem_text", "")
        workflow = state.get("workflow_type", "computational")
        start_time = datetime.now()

        query_vector = self.embeddings.embed_query(query)
        where_filter = {"source": db_value} if topic != "unknown" else None

        # Retrieval with Metadata Filtering
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=5,
            where=where_filter
        )
        # Format Raw Chunks for the UI (Source Tracking)
        raw_chunks = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                raw_chunks.append({
                    "id": f"Source {i+1}",
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })

        # Consolidation (The "Synthesizer")
        if not raw_chunks:
            consolidated = "No specific reference found in the knowledge base for this topic."
        else:
            consolidated = self._synthesize(query, raw_chunks, workflow)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "rag_chunks": raw_chunks,
            "consolidated_knowledge": consolidated,
            "rag_trace": [{
                "event": "retrieval_complete",
                "sources_found": len(raw_chunks),
                "topic_filter": topic,
                "execution_time_ms": execution_time
            }]
        }
    
    def _synthesize(self, query: str, chunks: List[Dict], workflow: str) -> str:
        """
        Internal LLM call to create the 'Cheat Sheet'.
        """
        # Prepare context with IDs for the LLM
        context_block = "\n\n".join([f"[{c['id']}]: {c['content']}" for c in chunks])

        # Tailor the synthesis based on the workflow
        if workflow == "conceptual":
            instruction = "Provide a clear, pedagogical definition and the primary formula."
        else:
            instruction = "Extract specific formulas, constants, and step-by-step methodology needed for calculation."

        prompt = f"""
        You are a Math Knowledge Librarian. Your goal is to provide the 'Cheat Sheet' for a Math Solver.

        USER PROBLEM: {query}

        RAW KNOWLEDGE CHUNKS:
        {context_block}

        TASK:
        {instruction}

        RULES:
        1. Use LaTeX for all math ($...$).
        2. You MUST cite the Source ID (e.g., [Source 1]) next to every formula or rule you extract.
        3. Do NOT solve the problem. Only provide the tools/facts needed to solve it.
        4. If a 'Common Pitfall' or 'Mistake' is mentioned in the chunks, include a 'WARNING' section.

        CONSOLIDATED CHEAT SHEET:
        """
        
        response = self.llm.invoke(prompt)
        return response.content
