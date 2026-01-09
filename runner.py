import os
from dotenv import load_dotenv
from agents.graph import MathMentorGraph
from agents.state import create_initial_state
from dotenv import load_dotenv

load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
def run_live_mentor(query: str):
    MODEL_NAME = os.getenv("MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    CHROMA_PATH = "./chromadb"
    
    mentor_graph = MathMentorGraph(base_llm_name=MODEL_NAME, chroma_path=CHROMA_PATH)
    app = mentor_graph.workflow 
    initial_state = create_initial_state(query, session_id="live_test_003")

    print(f"\n{'='*30} STARTING LIVE FLOW {'='*30}")
    
    # We use stream to see the agents working in real-time
    for event in app.stream(initial_state):
        for node_name, output in event.items():
            print(f"\n[NODE]: {node_name.upper()}")
            
            if node_name == "parser":
                print(f"   🎯 Intent: {output.get('intent')} | Topic: {output.get('topic')}")
            
            elif node_name == "router":
                print(f"   🛤️  Workflow: {output.get('workflow_type')}")
                print(f"   📋 Plan: {', '.join(output.get('plan', []))}")
            
            elif node_name == "rag":
                print(f"   📚 Sources Found: {len(output.get('rag_chunks', []))}")
            
            elif node_name == "solver":
                # This is where the 'Thinking' happens
                trace = output.get("solver_trace", [])
                for step in trace:
                    if "plan" in step:
                        print(f"   🧠 Thought: {step['plan']}")
                    if "code" in step:
                        print(f"   💻 Python Code:\n{step['code']}")
                    if "result" in step:
                        print(f"   ✅ Execution Result: {step['result']}")

            elif node_name == "verifier":
                trace = output.get("verifier_trace", [])
                for step in trace:
                    if "code" in step:
                        print(f"   ⚖️  Verification Code:\n{step['code']}")
                    if "result" in step:
                        print(f"   🔍 Audit Result: {step['result']}")
                
                status = "✅ VERIFIED" if output.get("is_correct") else "❌ REJECTED"
                print(f"   🏁 Decision: {status}")

            elif node_name == "explainer":
                print(f"   🎓 Explainer finished generating the lesson.")

   # FINAL OUTPUT
    final_state = app.invoke(initial_state) 
    
    print(f"\n{'='*80}")
    print("FINAL MENTOR RESPONSE:")
    
    # SENIOR LOGIC: Check which field contains the response
    if final_state.get("direct_response"):
        # This was a Chitchat or Out-of-Context response from the Parser
        print(final_state["direct_response"])
    elif final_state.get("explanation"):
        # This was a Math solution from the Explainer
        print(final_state["explanation"])
    else:
        print("No response was generated. Check the agent traces for errors.")
        
    if final_state.get("key_concepts"):
        print(f"\nKey Concepts: {', '.join(final_state['key_concepts'])}")
        
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # test_query = "If a bag has 4 red and 6 black balls, what is the probability of picking 2 red balls without replacement?"
    test_query = "Explain me Baye's theorem in probability?"
    run_live_mentor(test_query)