## Project Structure

### Directory Layout
```
math-mentor/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── app.py                              # Main Streamlit application
│
├── agents/
│   ├── __init__.py
│   ├── graph.py                        # LangGraph workflow definition
│   ├── parser_agent.py                 # Question parsing
│   ├── router_agent.py                 # Workflow routing
│   ├── rag_agent.py                    # Retrieval agent
│   ├── solver_agent.py                 # Problem solving
│   ├── verifier_agent.py               # Solution verification
│   ├── explainer_agent.py              # Explanation generation
│   └── hitl_agent.py                   # Human-in-the-loop
│
├── rag/
│   ├── __init__.py
│   ├── embeddings.py                   # Embedding generation
│   ├── retriever.py                    # ChromaDB retrieval
│   └── build_knowledge_base.py         # KB building script
│
├── memory/
│   ├── __init__.py
│   ├── store.py                        # Memory storage
│   └── session_manager.py              # Session state
│
├── utils/
│   ├── __init__.py
│   ├── llm.py                          # LLM management
│   ├── prompts.py                      # Agent prompts
│   ├── helpers.py                      # Utility functions
│   └── validators.py                   # Input validation
│
├── knowledge_base/
│   └── probability/
│       ├── 01_foundations.md
│       ├── 02_conditional_probability.md
│       ├── 03_bayes_theorem.md
│       ├── 04_counting_principles.md
│       ├── 05_distributions.md
│       ├── 06_problem_templates.md
│       ├── 07_common_mistakes.md
│       └── 08_verification_checklists.md
│
├── chroma_db/                          # Git-committed vector DB
│   ├── chroma.sqlite3
│   └── [UUID folders]
│
├── data/
│   └── memory.json                     # Session memory persistence
│
└── tests/
    ├── test_agents.py
    ├── test_rag.py
    └── test_memory.py