from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import logging 

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY missing")

LLM = ChatGroq(
    temperature = float(os.getenv("TEMPERATURE", 0.3))
                                  ,
    model_name = os.getenv("MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    groq_api_key = GROQ_API_KEY
)