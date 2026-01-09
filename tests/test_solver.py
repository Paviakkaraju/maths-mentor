from agents.solver_agent import SolverAgent
from langchain_groq import ChatGroq
from utils.llm import LLM
import os
from dotenv import load_dotenv
load_dotenv(override=True)

MODEL = os.getenv("MODEL")

llm = ChatGroq(model=MODEL, temperature=0.01)

# Initialize
solver = SolverAgent(llm, max_iterations=5, verbose=True)

# Test problem
state = {
    'problem_text': 'A bag has 4 red and 6 black balls. Find the probability of drawing 2 red balls without replacement.',
    'variables': [],
    'constraints': ['without replacement'],
    'consolidated_knowledge': """
            The probability of an event 
𝐸
E is defined as

𝑃
(
𝐸
)
=
𝑛
(
𝐸
)
𝑛
(
𝑆
)
P(E)=
n(S)
n(E)
​

,
where 
𝑛
(
𝐸
)
n(E) is the number of favourable outcomes and 
𝑛
(
𝑆
)
n(S) is the total number of possible outcomes.

The combination formula, which is used to count selections where order does not matter, is

𝑛
𝐶
𝑟
=
𝑛
!
𝑟
!
(
𝑛
−
𝑟
)
!
n
C
r
​

=
r!(n−r)!
n!
​

,
where 
𝑛
!
n! denotes the factorial of 
𝑛
n.

For two events 
𝐴
A and 
𝐵
B, the probability of their union is given by

𝑃
(
𝐴
∪
𝐵
)
=
𝑃
(
𝐴
)
+
𝑃
(
𝐵
)
−
𝑃
(
𝐴
∩
𝐵
)
P(A∪B)=P(A)+P(B)−P(A∩B).
If the events 
𝐴
A and 
𝐵
B are mutually exclusive, then the formula simplifies to

𝑃
(
𝐴
∪
𝐵
)
=
𝑃
(
𝐴
)
+
𝑃
(
𝐵
)
P(A∪B)=P(A)+P(B).

When drawing objects such as balls or cards without replacement, if there are 
𝑛
n objects in total and 
𝑟
r objects are drawn, the total number of possible outcomes is 
𝑛
𝐶
𝑟
n
C
r
​

. If among the total 
𝑛
n objects, 
𝑟
r objects are of a particular type, the probability of drawing one object of that type in a single draw is 
𝑟
𝑛
n
r
​

.

In problems involving drawing cards or balls without replacement, such as selecting a specific combination of objects, the probability is calculated by finding the number of favourable combinations and dividing it by the total number of possible combinations using the formula 
𝑃
(
𝐸
)
=
𝑛
(
𝐸
)
𝑛
(
𝑆
)
P(E)=
n(S)
n(E)
​

.

When solving such problems, it is important to remember that without replacement, the total number of outcomes and the number of favourable outcomes change with each draw. Failing to update these values correctly leads to incorrect probability calculations.

For example, if a bag contains 4 red balls and 6 black balls, the total number of balls is 10, and the number of red balls is 4. These facts, along with the formulas above, are sufficient to solve problems such as finding the probability of drawing 2 red balls without replacement.
                                """
}

# Solve
result = solver.solve(state)

# Print results
print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(f"Answer: {result['final_answer']}")
print(f"Time: {result['execution_time_ms']}ms")
print(f"Iterations: {result['iteration_count']}")
print(f"Success: {result['success']}")
print("\nTrace:")
for i, entry in enumerate(result['solver_trace']):
    print(f"\n{i+1}. {entry}")