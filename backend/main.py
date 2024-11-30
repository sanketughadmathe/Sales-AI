import os

from app.routes import chat
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI(title="Sales Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api")


@app.get("/")
async def read_root():
    return {"message": "Welcome to Sales Assistant API"}


# Add environment check endpoint
@app.get("/check-env")
async def check_environment():
    """Endpoint to verify environment variables are loaded correctly"""
    return {
        "vespa_url": os.getenv("VESPA_URL") is not None,
        "vespa_cert": os.getenv("VESPA_CERT_PATH") is not None,
        "vespa_key": os.getenv("VESPA_KEY_PATH") is not None,
        "anthropic_key": os.getenv("ANTHROPIC_API_KEY") is not None,
    }
