import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq


from agents.graph import MathMentorGraph
from agents.state import create_initial_state
from utils.image_processor import ImageProcessor
from utils.audio_processor import AudioProcessor


# 1. Page Config
st.set_page_config(page_title="AI Math Mentor", page_icon="🎓", layout="wide")
load_dotenv()
model_name = os.getenv("MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


# Custom CSS for better input styling
st.markdown("""
    <style>
    /* Make chat input container have padding */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
    /* Container for input and buttons */
    .input-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }
    
    /* Make the input take most space */
    .input-row > div:first-child {
        flex: 1;
    }
    
    /* Button container */
    .button-group {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    
    /* Style individual action buttons */
    div[data-testid="column"] button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    div[data-testid="column"] button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    /* Hide button labels */
    div[data-testid="column"] button p {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_image_processor():
    """Initialize and cache the image processor"""
    return ImageProcessor()

@st.cache_resource
def get_audio_processor():
    """Initialize and cache the audio processor"""
    return AudioProcessor()

@st.cache_resource
def get_graph():
    """Initialize and cache the mentor graph"""
    llm = ChatGroq(model=model_name, temperature=0)
    return MathMentorGraph(base_llm_name=model_name, chroma_path="./chromadb")


try:
    mentor_app = get_graph()
except Exception as e:
    st.error(f"Failed to initialize mentor: {str(e)}")
    st.stop()


# 3. UI Header
st.title("🎓 AI Math Mentor")
st.markdown("I help you solve and understand JEE-level Math problems.")


# 4. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "show_ocr_editor" not in st.session_state:
    st.session_state.show_ocr_editor = False
if "processing_ocr" not in st.session_state:
    st.session_state.processing_ocr = False
if "audio_transcript" not in st.session_state:
    st.session_state.audio_transcript = ""
if "show_audio_editor" not in st.session_state:
    st.session_state.show_audio_editor = False
if "processing_audio" not in st.session_state:
    st.session_state.processing_audio = False
if "final_query" not in st.session_state:
    st.session_state.final_query = ""
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Show uploaded image if present
if st.session_state.uploaded_image:
    with st.expander("📎 Uploaded Image", expanded=False):
        st.image(st.session_state.uploaded_image, width=300)
        if st.button("❌ Remove Image"):
            st.session_state.uploaded_image = None
            st.rerun()


# Image upload dialog (appears when button is clicked)
if st.session_state.get("show_image_upload", False):
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload a math problem image", 
            type=["png", "jpg", "jpeg"],
            key="file_uploader"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Upload", type="primary"):
                if uploaded_file:
                    st.session_state.uploaded_image = uploaded_file
                    
                    # Process Image
                    with st.spinner("🔍 Extracting text from image..."):
                        try:
                            processor = get_image_processor()
                            image_bytes = uploaded_file.getvalue()
                            result = processor.extract_text(image_bytes)
                            result["success"] = True
                        except Exception as e:
                            st.error(f"Failed to process image: {str(e)}")
                            result = {"extracted_text": "", "confidence": 0.0, "success": False}
                    
                    if result.get("success", False):
                        st.session_state.ocr_text = result["extracted_text"]
                        st.session_state.ocr_confidence = result["confidence"]
                        st.session_state.show_ocr_editor = True
                        st.session_state.show_image_upload = False
                        st.rerun()
                    else:
                        st.error("Could not extract text from image. Please try another image.")
                else:
                    st.warning("Please select an image first")
        
        with col_b:
            if st.button("Cancel"):
                st.session_state.show_image_upload = False
                st.rerun()


# Audio Recorder Logic
if st.session_state.get("show_audio_recorder", False):
    with st.container():
        audio_value = st.audio_input("Record your math question")
        
        if audio_value:
             # Process Audio
             with st.spinner("🎧 Transcribing audio..."):
                 try:
                     processor = get_audio_processor()
                     audio_bytes = audio_value.read()
                     result = processor.transcribe(audio_bytes)
                 except Exception as e:
                     st.error(f"Failed to process audio: {str(e)}")
                     result = {"transcript": "", "success": False}
             
             if result.get("success", False):
                 st.session_state.audio_transcript = result["transcript"]
                 st.session_state.show_audio_editor = True
                 st.session_state.show_audio_recorder = False
                 st.rerun()
             else:
                 st.error(f"Audio transcription failed: {result.get('error', 'Unknown error')}")
        
        if st.button("Cancel Audio"):
            st.session_state.show_audio_recorder = False
            st.rerun()


# HITL Component: OCR Verification
if st.session_state.show_ocr_editor:
    with st.container():
        st.info("📝 Please review the extracted text from your image:")
        
        if st.session_state.ocr_confidence < 0.7:
            st.warning(f"⚠️ Low OCR Confidence: {st.session_state.ocr_confidence}. Please verify the text carefully.")
            
        edited_text = st.text_area("Extracted Text", value=st.session_state.ocr_text, height=150)
        
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            if st.button("🚀 Confirm & Solve", type="primary"):
                st.session_state.final_query = edited_text
                st.session_state.processing_ocr = True
                st.session_state.show_ocr_editor = False
                st.rerun()
                
        with col2:
            if st.button("Cancel"):
                st.session_state.show_ocr_editor = False
                st.session_state.uploaded_image = None
                st.rerun()


# HITL Component: Audio Verification
if st.session_state.show_audio_editor:
    with st.container():
        st.info("🎤 Please review the transcribed text:")
        
        edited_transcript = st.text_area("Transcribed Text", value=st.session_state.audio_transcript, height=150)
        
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            if st.button("🚀 Confirm & Solve", type="primary", key="confirm_audio"):
                st.session_state.final_query = edited_transcript
                st.session_state.processing_audio = True
                st.session_state.show_audio_editor = False
                st.rerun()
                
        with col2:
            if st.button("Cancel", key="cancel_audio"):
                st.session_state.show_audio_editor = False
                st.rerun()


# 5. Input Section with Action Buttons ALIGNED RIGHT
# Create container for input row
input_container = st.container()

with input_container:
    # Create columns: input area (wide) + buttons (narrow)
    input_col, button_col = st.columns([0.70, 0.1])
    
    with input_col:
        prompt = st.chat_input("Type your math question here...")
    
    with button_col:
        # Create two columns for the two buttons side by side
        mic_col, img_col = st.columns(2)
        
        with mic_col:
            if st.button("🎤", help="Voice Input", key="voice_btn"):
                st.session_state.show_audio_recorder = True
                st.rerun()
        
        with img_col:
            if st.button("📷", help="Upload Image", key="img_btn"):
                st.session_state.show_image_upload = True


# 6. Execution Logic
user_input = None
input_mode = "text"

if prompt:
    user_input = prompt

if st.session_state.processing_ocr:
    user_input = st.session_state.final_query
    input_mode = "image"
    st.session_state.processing_ocr = False

if st.session_state.processing_audio:
    user_input = st.session_state.final_query
    input_mode = "audio"
    st.session_state.processing_audio = False

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):
        try:
            # Create initial state
            current_state = create_initial_state(
                user_input, 
                session_id="streamlit_session", 
                input_mode=input_mode
            )
            
            with st.status("👨‍🏫 Mentor is thinking...", expanded=True) as status:
                # Stream the graph execution
                for event in mentor_app.workflow.stream(current_state):
                    # Update state with event data
                    for node_name, node_output in event.items():
                        current_state.update(node_output)
                        
                        # UI Trace Logic
                        if node_name == "parser":
                            intent = current_state.get('intent', 'unknown')
                            topic = current_state.get('topic', 'unknown')
                            st.write(f"🔍 **Parser:** Intent `{intent}` | Topic `{topic}`")
                            
                            if st.session_state.show_debug:
                                with st.expander("Parser Details"):
                                    st.json({
                                        "intent": intent,
                                        "topic": topic,
                                        "variables": current_state.get('variables', []),
                                        "is_valid": current_state.get('is_valid', False)
                                    })
                        
                        elif node_name == "router":
                            workflow = current_state.get('workflow_type', 'unknown')
                            st.write(f"🛤️ **Router:** Using `{workflow}` workflow")
                        
                        elif node_name == "rag":
                            chunks = current_state.get('rag_chunks', [])
                            st.write(f"📚 **RAG:** Found {len(chunks)} sources")
                            
                            if st.session_state.show_debug and chunks:
                                with st.expander("RAG Sources"):
                                    for i, chunk in enumerate(chunks[:3], 1):
                                        st.caption(f"Source {i}: {chunk[:200]}...")
                        
                        elif node_name == "solver":
                            st.write("🧮 **Solver:** Calculating...")
                            
                            solver_trace = current_state.get("solver_trace", [])
                            if solver_trace:
                                last_step = solver_trace[-1]
                                
                                # Show code if present
                                if "code" in last_step:
                                    with st.expander("Code Executed", expanded=False):
                                        st.code(last_step["code"], language="python")
                                
                                # Show result if present
                                if "result" in last_step:
                                    st.caption(f"Result: {last_step['result'][:100]}")
                        
                        elif node_name == "verifier":
                            is_correct = current_state.get("is_correct", False)
                            status_icon = "✅" if is_correct else "❌"
                            st.write(f"{status_icon} **Verifier:** Audit complete")
                            
                            if st.session_state.show_debug:
                                with st.expander("Verification Details"):
                                    st.json({
                                        "is_correct": is_correct,
                                        "confidence": current_state.get("verification_confidence", 0)
                                    })


                status.update(label="✅ Analysis Complete", state="complete", expanded=False)


            # FINAL DISPLAY LOGIC
            response_text = current_state.get("explanation") or current_state.get("direct_response")
            
            if response_text:
                st.markdown("### 🎓 Mentor's Explanation")
                st.markdown(response_text)
                
                # Show step-by-step breakdown if available
                step_explanations = current_state.get("step_explanations")
                if step_explanations and isinstance(step_explanations, list) and len(step_explanations) > 0:
                    st.markdown("### 📝 Step-by-Step Derivation")
                    for i, step in enumerate(step_explanations):
                        st.markdown(f"**{i+1}.** {step}")

                # Show final answer if available
                if current_state.get("final_answer"):
                    st.success(f"**Final Answer:** {current_state['final_answer']}")
                
                # Show key concepts
                if current_state.get("key_concepts"):
                    st.write("---")
                    st.caption("**Key Concepts:**")
                    concepts = current_state["key_concepts"]
                    st.write(" • ".join([f"`{c}`" for c in concepts]))
                
                # Show sources if available
                if current_state.get("sources") and st.session_state.show_debug:
                    with st.exponder("📚 Sources Used"):
                        for source in current_state["sources"]:
                            st.caption(f"• {source}")
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text
                })
                
            else:
                # More specific error messages
                if current_state.get("intent") == "chitchat":
                    error_msg = "This seems like a general conversation. I'm specialized in JEE Math problems. Try asking me to solve a problem or explain a concept!"
                elif current_state.get("intent") == "out_of_context":
                    error_msg = "This question is outside my expertise. I focus on JEE-level Math (probability, algebra, calculus, linear algebra)."
                elif not current_state.get("is_valid"):
                    error_msg = "I couldn't understand the problem. Could you rephrase it or provide more details?"
                else:
                    error_msg = "I processed the request but couldn't generate an explanation. This might be a system error."
                
                st.error(error_msg)
                
                if st.session_state.show_debug:
                    with st.expander("🐛 Debug Info"):
                        st.json({
                            "intent": current_state.get("intent"),
                            "topic": current_state.get("topic"),
                            "is_valid": current_state.get("is_valid"),
                            "workflow_type": current_state.get("workflow_type"),
                            "has_explanation": bool(current_state.get("explanation")),
                            "has_direct_response": bool(current_state.get("direct_response"))
                        })
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            
            if st.session_state.show_debug:
                st.exception(e)
            
            # Suggest solutions
            st.info("""
            **Possible solutions:**
            1. Check if all environment variables are set (GROQ_API_KEY, etc.)
            2. Verify the graph workflow is properly configured
            3. Try a simpler question first
            4. Check the debug toggle in the sidebar for more details
            """)


# Footer
st.markdown("---")
st.caption("💡 Tip: You can ask me to solve problems or explain concepts!")
