# Math Mentor: A Reliable Multimodal Multi-Agent System
Math Mentor is an end-to-end AI application designed to solve and explain JEE-style mathematics problems with 100% arithmetic reliability. Unlike standard LLMs that "guess" math results, Math Mentor uses a Multi-Agent Orchestration that reasons, writes code, and self-corrects through an independent audit loop.


## Key Features
- Multimodal Input: Supports Text, Image (OCR via EasyOCR), and Audio (ASR via Groq Whisper).
- Agentic Workflow: Orchestrated by LangGraph, featuring a team of specialized agents.
- Zero-Hallucination Math: A Solver Agent that executes symbolic math in a Hardened Python REPL (SymPy/NumPy).
- Self-Correction Loop: A Verifier Agent that independently audits the solver's logic and triggers autonomous retries if errors are detected.
- Agentic RAG: Retrieves and "consolidates" math formulas into high-density context for the agents.
- Human-in-the-Loop (HITL): Mandatory review step for OCR/ASR extractions to ensure "Garbage In, Garbage Out" is avoided.
- Pedagogical Explanations: An Explainer Agent that provides student-friendly lessons with analogies and key concepts.

## Architecture: The Multi-Agent Brain
The system is built on a **StateGraph** where data flows through specialized nodes:
- **Parser Agent:** Cleans input and identifies mathematical intent.
- **Intent Router:** Categorizes queries into conceptual (definitions) or computational (solving) paths.
- **RAG Agent:** Performs semantic search on a curated knowledge base and synthesizes a "Cheat Sheet."
- **Solver Agent (ReAct):** Plans the solution and executes Python code.
- **Verifier Agent:** The "Critic" that performs a reverse-check on the solver's result.
- **Explainer Agent:** The "Tutor" that formats the final pedagogical output.
The Self-Correction Loop
If the Verifier rejects the Solver's answer, it appends a Mentor Note to the context and routes the state back to the Solver for a second attempt (Attempt 2).

## Tech Stack
- **LLM Inference:** Groq (Llama-3.1-70B & Llama-4-Scout)
- **Orchestration:** LangGraph
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace all-mpnet-base-v2
- **OCR:** EasyOCR
- **ASR:** Groq Whisper-large-v3
- **Frontend:** Streamlit

## Security: Hardened Python REPL
To ensure the system is production-ready, the Python execution environment is strictly sandboxed:
- **AST Whitelisting:** Only allows mathematical nodes (BinOp, Call, Assign). Blocks Import, Attribute (prevents __dict__ attacks), and Subscript.
- **Process Isolation:** Every calculation runs in a separate multiprocessing.
- **Resource Guards:** 5-second hard timeout and output size limits to prevent DoS attacks.

## Usage
- Text: Type your math problem directly into the chat.
- Image: Click the 📷 button to upload a photo/screenshot. Review the OCR text and click "Confirm & Solve."
- Audio: Click the 🎤 icon to speak your question. Review the transcript and click "Confirm & Solve."
- Reasoning: Expand the "Mentor is thinking..." status box to see the full Chain of Thought (CoT), including the Python code executed by the agents.

### Developed by Pavithra A
Built for the AI Engineer Assignment at AI Planet - Reliable Multimodal Math Mentor.
