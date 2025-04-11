from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import logging
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from typing import Optional, Dict, Any
import signal
import sys

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Debug environment variables
logging.info("Checking environment variables...")
logging.info(f"AGENT_ENDPOINT: {'Set' if os.getenv('AGENT_ENDPOINT') else 'Not Set'}")
logging.info(f"AGENT_KEY: {'Set' if os.getenv('AGENT_KEY') else 'Not Set'}")
logging.info(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set'}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str

class ChatLogsResponse(BaseModel):
    logs: Dict[str, Dict[str, Any]]

# In-memory cache for chat logs
chat_logs = {}

async def load_chat_log():
    """Load existing chat log from JSON file"""
    global chat_logs
    log_file = "data/chat_log.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                chat_logs = json.load(f)
        except json.JSONDecodeError:
            chat_logs = {}
    return chat_logs

async def save_chat_log_async():
    """Save chat log to JSON file asynchronously"""
    log_file = "data/chat_log.json"
    try:
        with open(log_file, 'w') as f:
            json.dump(chat_logs, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving chat log: {str(e)}")

def get_openai_response(text: str) -> Optional[str]:
    """Get response from OpenAI API"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    logging.info(f"OpenAI API key length: {len(openai_api_key) if openai_api_key else 0}")
    
    if not openai_api_key:
        logging.error("Missing OpenAI API key")
        return None

    try:
        client = OpenAI(api_key=openai_api_key)
        logging.info("OpenAI client created successfully")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": text}
            ]
        )
        logging.info("OpenAI response received successfully")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return None

def get_deepseek_response(text: str) -> Optional[str]:
    """Get response from Deepseek API"""
    agent_endpoint = os.getenv("AGENT_ENDPOINT")
    agent_key = os.getenv("AGENT_KEY")
    
    if not agent_endpoint or not agent_key:
        logging.error("Missing Deepseek API configuration")
        return None

    try:
        client = OpenAI(
            base_url=agent_endpoint,
            api_key=agent_key,
        )
        
        response = client.chat.completions.create(
            model="n/a",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Deepseek API error: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load chat logs when the application starts"""
    await load_chat_log()

@app.on_event("shutdown")
async def shutdown_event():
    """Save chat logs when the application shuts down"""
    await save_chat_log_async()
    logging.info("Server shutdown complete")

@app.get("/logs", response_model=ChatLogsResponse)
async def get_logs():
    """Get all chat logs"""
    try:
        # Ensure we have the latest data
        await load_chat_log()
        return ChatLogsResponse(logs=chat_logs)
    except Exception as e:
        logging.error(f"Error retrieving logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logging.info(f"Received request with text: {request.text}")
        
        if not request.text:
            raise HTTPException(status_code=400, detail="No text provided")

        # First try OpenAI
        logging.info("Attempting to get response from OpenAI")
        response = get_openai_response(request.text)
        source = "openai"
        
        # If OpenAI fails, fall back to Deepseek
        if response is None:
            logging.info("OpenAI failed, falling back to Deepseek")
            response = get_deepseek_response(request.text)
            source = "deepseek"
        
        if response is None:
            logging.error("Both OpenAI and Deepseek failed to provide a response")
            raise HTTPException(status_code=500, detail="Failed to get response from both APIs")

        # Log the interaction in memory
        timestamp = datetime.now().isoformat()
        chat_logs[timestamp] = {
            "input": request.text,
            "output": response,
            "source": source
        }

        # Save to file asynchronously without blocking
        asyncio.create_task(save_chat_log_async())

        return ChatResponse(response=response)

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logging.info("Shutdown signal received")
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("Starting server...")
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=9003,
            log_level="info",
            timeout_graceful_shutdown=5  # Shutdown timeout in seconds
        )
    except KeyboardInterrupt:
        logging.info("Server shutdown initiated by user")
        sys.exit(0)
