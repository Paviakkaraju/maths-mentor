import os
from pprint import pprint
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.rag_agent import RAGAgent
from agents.state import create_initial_state

# Load environment variables
load_dotenv()

def test_rag_flow():
    # Initialize the LLM
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        temperature=0.3
    )

    # Initialize the RAG Agent
    rag_agent = RAGAgent(llm, chroma_path="./chromadb")

    # MOCK STATE 
    # user_query = "A bag has 4 red and 6 black balls. Find the probability of drawing 2 red balls without replacement."
    user_query = "What is conditional probability"

    state = create_initial_state(user_query, session_id="test_123")
    
    # Manually fill the fields the previous agents would have filled
    state["problem_text"] = user_query
    state["topic"] = "probability"
    # state["workflow_type"] = "computational" 
    state["workflow_type"] = "contextual"

    print("="*50)
    print(f"TESTING RAG AGENT")
    print(f"Query: {user_query}")
    print(f"Topic Filter: {state['topic']}")
    print("="*50)

    # RUN THE AGENT
    try:
        updated_state_patch = rag_agent.retrieve_and_consolidate(state)
        
        # Manually apply the patch to our state for inspection
        state.update(updated_state_patch)

        # INSPECT THE RESULTS
        print("\n[1] RAW CHUNKS RETRIEVED:")
        if not state["rag_chunks"]:
            print("No chunks found. Check if your ChromaDB is populated and the topic matches.")
        else:
            for chunk in state["rag_chunks"]:
                print(f"- {chunk['id']}: {chunk['content'][:100]}...")

        print("\n[2] CONSOLIDATED KNOWLEDGE (THE CHEAT SHEET):")
        print("-" * 30)
        print(state["consolidated_knowledge"])
        print("-" * 30)

        print("\n[3] RAG TRACE:")
        pprint(state["rag_trace"])

    except Exception as e:
        print(f"Execution Failed: {e}")

if __name__ == "__main__":
    test_rag_flow()