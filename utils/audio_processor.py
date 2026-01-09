import os
import json
from dotenv import load_dotenv

# Load API Key
load_dotenv()

import os
from groq import Groq

class AudioProcessor:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def transcribe(self, audio_bytes) -> dict:
        """
        Transcribes audio using Groq's Whisper-large-v3.
        """
        try:
            # Save bytes to a temporary file (Whisper API requires a file object)
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_bytes)
            
            with open("temp_audio.wav", "rb") as file:
                # The 'prompt' helps Whisper understand math context
                transcription = self.client.audio.transcriptions.create(
                    file=("temp_audio.wav", file.read()),
                    model="whisper-large-v3",
                    prompt="The audio contains math equations, symbols like square root, integral, and variables like x, y, z.",
                    response_format="json",
                    language="en"
                )
            
            return {
                "transcript": transcription.text,
                "success": True
            }
        except Exception as e:
            return {"transcript": "", "success": False, "error": str(e)}
        finally:
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")

def run_audio_test(file_path: str):
    print(f"\n{'='*60}")
    print(f"🎤 TESTING AUDIO PROCESSOR")
    print(f"File: {file_path}")
    print(f"{'='*60}")

    # 1. Initialize Processor
    processor = AudioProcessor()

    # 2. Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found.")
        return

    # 3. Read bytes and Transcribe
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        print("⏳ Transcribing via Groq Whisper-v3...")
        result = processor.transcribe(audio_bytes)

        if result["success"]:
            transcript = result["transcript"]
            print(f"\n✅ TRANSCRIPT: \n\"{transcript}\"")
            
            # 4. Math-Specific Phrase Detection (Assignment Requirement 1.B)
            # We check if the transcription correctly captured technical terms
            math_keywords = ["square root", "integral", "limit", "squared", "plus", "minus", "divided by"]
            found_keywords = [word for word in math_keywords if word in transcript.lower()]
            
            print(f"\n📊 MATH KEYWORDS DETECTED: {found_keywords}")
            
            # 5. Success Check
            if len(transcript) > 5:
                print("\n✨ TEST PASSED: Audio successfully converted to text.")
            else:
                print("\n⚠️ TEST WARNING: Transcript is very short. Check audio quality.")
                
        else:
            print(f"\n❌ TRANSCRIPTION FAILED: {result.get('error')}")

    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {str(e)}")

if __name__ == "__main__":
    # Change this to your actual test file name
    TEST_FILE = "math_test.wav" 
    
    # Create a dummy check if file doesn't exist
    if not os.path.exists(TEST_FILE):
        print(f"Please provide a sample audio file named '{TEST_FILE}' to run this test.")
    else:
        run_audio_test(TEST_FILE)