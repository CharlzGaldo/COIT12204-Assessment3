"""
Priority-classification agent.

Flow:
    1. Run the `check_urgency_keywords` tool against the task text.
    2. Feed the task + tool result into a Gemini chat model configured
       for structured output (Pydantic schema).
    3. Update AgentState with the tool call, the result, or the error.
    4. Fall back to a safe default classification if the LLM call fails.

Keeping the LLM as an injectable parameter (rather than a module-level
singleton called directly) is what makes this testable with monkeypatch/
mocking: tests can pass in a fake object with the same `.invoke()` shape.
"""
import logging
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import settings
from app.state import AgentState, agent_state
from app.tools import check_urgency_keywords

# Named logger so log lines from this module are clearly tagged in the
# console output (see logging.basicConfig() in main.py).
logger = logging.getLogger("agent.core")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
# This is the contract between the LLM and the rest of the app. Passing this
# class to `model.with_structured_output(...)` forces Gemini's response to
# be parsed into exactly these two fields, instead of us having to parse
# free-form text ourselves (which would be fragile).
class PriorityClassification(BaseModel):
    """Structured output schema the LLM must return."""

    # Literal[...] restricts the value to exactly one of these three strings.
    # If the model tries to return anything else, pydantic will raise a
    # validation error, which we catch further down and turn into a fallback.
    priority: Literal["High", "Medium", "Low"] = Field(
        description="The classified priority level of the task."
    )
    reasoning: str = Field(
        description="A short explanation for why this priority was chosen."
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# ChatPromptTemplate lets us define the conversation shape once, with
# {placeholders}, and fill them in at call time via .invoke({...}).
# "system" = instructions/persona, "human" = the actual user-turn content.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a task-triage assistant. Classify the priority of the "
            "given task as High, Medium, or Low, and briefly justify your "
            "choice. Use the urgency-keyword scan result as supporting "
            "evidence, but weigh the overall meaning of the task too. "
            "If the task description itself explicitly states its own "
            "priority or urgency level (e.g. 'medium urgency', 'low "
            "priority', 'not urgent'), treat that stated level as a strong "
            "signal and follow it unless the rest of the description "
            "clearly contradicts it — do not default to a higher priority "
            "just because a word like 'urgency' appears in the text.",
        ),
        (
            "human",
            "Task description: {task_description}\n\n"
            "Urgency keyword scan result: {tool_result}",
        ),
    ]
)

# Safe default returned whenever the LLM call fails for any reason (bad key,
# network issue, rate limit, malformed response, etc). This is what keeps
# the /classify_priority endpoint from ever hard-crashing on an LLM problem.
FALLBACK_RESULT = {
    "priority": "Medium",
    "reasoning": "Fallback classification: the agent could not reach the "
    "language model, so a safe default of Medium was used.",
}


def get_llm() -> ChatGoogleGenerativeAI:
    """Construct the real Gemini chat model. Not called during tests.

    Tests never call this directly — they either pass their own fake `llm`
    object into classify_priority(), or monkeypatch this function itself.
    That's what keeps the test suite from needing a real API key.
    """
    if not settings.google_api_key:
        # Fail loudly and early with a clear message rather than letting
        # the Gemini SDK raise a more cryptic auth error later.
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        # temperature=0 makes classification more deterministic/repeatable,
        # which matters for an agent whose job is consistent triage rather
        # than creative variation.
        temperature=0,
    )


def classify_priority(
    task_description: str,
    llm: Optional[ChatGoogleGenerativeAI] = None,
    state: AgentState = agent_state,
) -> dict:
    """Run the agent end-to-end for a single task description.

    Args:
        task_description: The task text to classify.
        llm: Optional pre-built chat model (used by tests to inject a mock).
             When None, the real Gemini model is built via get_llm().
        state: The AgentState instance to update (defaults to the shared
               module-level instance used by the FastAPI app).

    Returns:
        A dict matching PriorityClassification, plus a "fallback" flag
        that's True whenever the safe default had to be used.
    """
    # --- Input validation -------------------------------------------------
    # Required feature: "Input validation" + "Handle errors gracefully".
    # We validate here (not just in the FastAPI layer) so classify_priority
    # is safe to call directly too, e.g. from a script or another endpoint.
    if not task_description or not task_description.strip():
        raise ValueError("task_description must not be empty")

    try:
        # --- Step 1: Tool call --------------------------------------------
        # .invoke() is the standard LangChain way to call a @tool-decorated
        # function. We always run this tool up front (rather than letting
        # the LLM decide whether to call it) to keep the flow simple and
        # deterministic for a single-agent assessment.
        tool_result = check_urgency_keywords.invoke({"task_description": task_description})

        # Record the tool call in state immediately — even if the LLM step
        # below fails, we still want a log of what the tool found.
        state.record_tool_call("check_urgency_keywords", tool_result)

        # --- Step 2: Structured LLM call -----------------------------------
        # Use the injected mock LLM if given (tests), otherwise build the
        # real Gemini client.
        model = llm or get_llm()

        # .with_structured_output(...) wraps the model so its response is
        # parsed straight into a PriorityClassification instance instead of
        # raw text we'd have to parse ourselves.
        structured_model = model.with_structured_output(PriorityClassification)

        # The `|` operator chains Runnables together: PROMPT formats the
        # messages, then structured_model consumes them. This is LangChain's
        # "LCEL" (LangChain Expression Language) chain syntax.
        chain = PROMPT | structured_model

        # .invoke() fills in the {task_description} / {tool_result}
        # placeholders and runs the whole chain in one call.
        result: PriorityClassification = chain.invoke(
            {"task_description": task_description, "tool_result": tool_result}
        )

        # Convert the pydantic model back into a plain dict for the caller
        # (FastAPI response models, tests, etc. all expect plain dicts).
        output = result.model_dump()
        output["fallback"] = False

        # Record this as the agent's latest successful suggestion.
        state.record_success(output)
        return output

    except Exception as exc:  # noqa: BLE001 - agent must degrade gracefully
        # Broad except is intentional here: this is the agent's single
        # safety net. Whatever goes wrong (network error, bad API key,
        # malformed structured output, rate limiting...), we log it, record
        # it in state, and hand back a safe default instead of raising.
        logger.exception("Agent run failed, using fallback classification")
        state.record_error(str(exc))

        output = dict(FALLBACK_RESULT)
        output["fallback"] = True
        return output