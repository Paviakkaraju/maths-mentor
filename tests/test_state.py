from agents.state import create_initial_state

state = create_initial_state(
user_input="What is P(sum=7) with two dice?",
session_id="test_001"
)

print("\n State created successfully!")
print(f"Session ID: {state['session_id']}")
print(f"Raw input: {state['raw_input']}")
print(f"Timestamp: {state['timestamp']}")

# Test state modification
state["intent"] = "math_problem"
state["problem_text"] = "What is the probability of getting sum 7 with two dice?"
state["topic"] = "probability"

print(f"\n State modified successfully!")
print(f"Intent: {state['intent']}")
print(f"Problem: {state['problem_text']}")
print(f"Topic: {state['topic']}")