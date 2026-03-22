from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from Researcher.agent import root_agent
import os
from dotenv import load_dotenv

# -------------------------
# CONFIG
# -------------------------
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment")

APP_NAME = os.getenv("APP_NAME", "Researcher_App")

# -------------------------
# INIT
# -------------------------
app = FastAPI()

session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# -------------------------
# REQUEST SCHEMA
# -------------------------
class QueryRequest(BaseModel):
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = "default_session"
    query: str


# -------------------------
# AUTH DEPENDENCY
# -------------------------
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# -------------------------
# ROUTE
# -------------------------
@app.post("/query")
async def query_agent(
    req: QueryRequest,
    x_api_key: str = Header(...)
):
    # 🔐 Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Ensure session exists
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=req.user_id,
        session_id=req.session_id
    )

    # Wrap message
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=req.query)]
    )

    response_text = ""

    # Run agent
    async for event in runner.run_async(
        user_id=req.user_id,
        session_id=req.session_id,
        new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_text += part.text

    return {
        "response": response_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
