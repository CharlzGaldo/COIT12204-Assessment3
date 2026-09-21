"""
FastAPI application exposing the priority-classification agent.

Run with:  uvicorn app.main:app --reload
Then open: http://127.0.0.1:8000/   (frontend)
       or: http://127.0.0.1:8000/docs  (auto-generated Swagger UI)
"""
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.agent import classify_priority
from app.state import agent_state

# Configure root logging once, at app startup. Every module in this
# project (agent.core, agent.state, agent.api) gets its own named logger
# but they all inherit this basicConfig, so log lines all show up in the
# same uvicorn console output with level + message.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent.api")

app = FastAPI(title="Task Priority Classification Agent")

# --- Static frontend ---------------------------------------------------
# Serves anything in app/static/ under the /static/ URL path (not used
# directly by index.html right now since it has no separate JS/CSS files,
# but this is the standard place to add them later).
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_frontend() -> FileResponse:
    """Serve the simple HTML frontend for manually testing the agent."""
    return FileResponse(STATIC_DIR / "index.html")


# --- Request / response schemas -----------------------------------------
# Pydantic models here define FastAPI's automatic request validation and
# response shape. Required feature: "Input validation" + "Output formatting".
class ClassifyRequest(BaseModel):
    # min_length=1 already rejects a completely empty string; the
    # field_validator below additionally rejects whitespace-only strings
    # (e.g. "   ") which min_length alone would let through.
    task_description: str = Field(..., min_length=1, max_length=2000)

    @field_validator("task_description")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_description must not be blank")
        return v


class ClassifyResponse(BaseModel):
    priority: str
    reasoning: str
    fallback: bool  # True if the agent had to use its safe-default answer


# --- Main agent endpoint --------------------------------------------------
@app.post("/api/agent/classify_priority", response_model=ClassifyResponse)
def classify_priority_endpoint(payload: ClassifyRequest) -> ClassifyResponse:
    """
    Required feature: FastAPI endpoint integration.

    FastAPI validates `payload` against ClassifyRequest automatically
    (before this function even runs) — a missing or blank task_description
    never reaches this code, it gets a 422 response straight away.
    """
    # %.80s truncates long task descriptions in the log so one giant input
    # doesn't flood the console.
    logger.info("Received classify_priority request: %.80s", payload.task_description)

    try:
        # Delegate to the actual agent logic in app/agent.py. Note this
        # endpoint itself contains almost no agent logic — that separation
        # is what makes classify_priority() independently unit-testable
        # without spinning up FastAPI at all (see tests/test_agent.py).
        result = classify_priority(payload.task_description)

    except ValueError as exc:
        # classify_priority() raises ValueError for blank input (defensive
        # double-check — FastAPI's validator above should normally catch
        # this first). Map it to a 422 Unprocessable Entity.
        logger.warning("Validation error in agent call: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:  # noqa: BLE001
        # Anything else unexpected (should be rare, since classify_priority
        # itself already has a broad try/except with a fallback — this is
        # a last-resort safety net at the API layer).
        logger.exception("Unexpected error in agent call")
        raise HTTPException(status_code=500, detail="Agent failed to process request") from exc

    logger.info("classify_priority result: %s", result)
    return ClassifyResponse(**result)


# --- Debug / demo endpoint -------------------------------------------------
@app.get("/api/agent/state")
def get_state() -> dict:
    """Debug endpoint: expose current agent state (useful for the demo video).

    Shows last_suggestion, error_count, last_tool_call, and the full
    history of logged state transitions — this is your evidence trail
    for the "state management" and "evaluation" sections of the report.
    """
    return agent_state.to_dict()


# --- Health check -----------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Simple liveness check — useful for confirming the server is up."""
    return {"status": "ok"}
