import os
import logging

from dotenv import load_dotenv

load_dotenv()

# Logging 
logging.basicConfig(level=logging.INFO)

# API Keys 
DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CARTESIA_API_KEY   = os.getenv("CARTESIA_API_KEY")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")
HF_TOKEN           = os.getenv("HF_TOKEN")

# Agent settings 
MAX_MESSAGES = 10  