import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.core.main_router import router as main_router
from api.utils.logger import setup_logger
from api.utils.constants import SHARED
from contextlib import asynccontextmanager
from api.scu.router import router as scu_router
from api.scu.agent_rag import AgenticRAG
from supabase import create_client, Client
from api.scu.computer_use import AnthropicActor
from logging import getLogger

load_dotenv()
setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = getLogger("API")
    try:
        # The unique Supabase URL which is supplied when you create a new project in your project dashboard.
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        # The unique Supabase Key which is supplied when you create a new project in your project dashboard.
        SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        SHARED["supabase_client"] = supabase
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")

    try:
        SHARED["agentic_rag"] = AgenticRAG()
    except Exception as e:
        logger.error(f"Failed to initialize extraction_agent: {e}")

    try:
        SHARED["anthropic_actor"] = AnthropicActor(
            max_tokens=4096
        )
    except Exception as e:
        logger.error(f"Failed to initialize extraction_agent: {e}")

    yield  # Application is running

    # Clean up resources during shutdown
    SHARED.clear()
    print("Shutdown completed. Resources cleaned up.")

app = FastAPI(
    lifespan=lifespan,
    description="API",
    title="API",
    version="0.1.0",
    contact={
        "name": "API",
        "url": "https://supabase.com",
    },
)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://supabase.com",
        "http://localhost:3000",
        "http://localhost:8000",
        # Add other origins as needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(main_router)
app.include_router(scu_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
