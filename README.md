# Task Priority Classification Agent — COIT12204 Assessment 3

A single LangChain-based agent that classifies the priority (High / Medium / Low)
of a task description, exposed via a FastAPI endpoint, with internal state
tracking and a pytest suite covering agent, state, and endpoint behaviour.

# View the application on web (Recommended)
https://priority-agent-458025859336.australia-southeast1.run.app

# View the video demo

## Architecture at a glance

```
app/
  config.py   # settings (Gemini API key, model name, log level)
  state.py    # AgentState: last_suggestion, error_count, last_tool_call, history
  tools.py    # check_urgency_keywords tool
  agent.py    # prompt template + structured-output LLM call + fallback logic
  main.py     # FastAPI app and /api/agent/classify_priority endpoint
  static/index.html  # simple browser frontend for testing/demoing the agent
tests/
  test_state.py      # AgentState transitions
  test_agent.py       # agent wrapper, tool, fallback logic (LLM mocked)
  test_endpoints.py   # FastAPI TestClient tests (agent mocked)
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```

3. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Open the frontend in your browser:
   ```
   http://127.0.0.1:8000/
   ```
   Type a task description, click "Classify Priority", and watch the
   Agent State panel update below it — this is the easiest way to
   demo the agent for your video (input → output → logged state
   transitions, all visible on one screen).

   Alternatively, test via curl:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/agent/classify_priority \
     -H "Content-Type: application/json" \
     -d '{"task_description": "Server is down in production, fix ASAP"}'
   ```

5. Inspect agent state directly:
   ```bash
   curl http://127.0.0.1:8000/api/agent/state
   ```

## Running tests

```bash
pytest -v
```

All LLM calls are mocked in tests (no API key or network access required to run
the test suite).

## Notes for the technical report

- **Agent design**: `agent.py` runs the `check_urgency_keywords` tool first,
  then calls Gemini with `with_structured_output` bound to the
  `PriorityClassification` Pydantic model, guaranteeing a parseable
  `{priority, reasoning}` result.
- **State**: `AgentState` tracks 4 variables (last_suggestion, error_count,
  last_tool_call, history) and logs every transition via the `logging` module.
- **Error handling**: any exception during the LLM call is caught, logged,
  recorded in state (`error_count` incremented), and a safe fallback
  classification of "Medium" is returned instead of crashing the endpoint.
- **AI assistance declaration**: this scaffold was drafted with Claude
  (Anthropic) assistance; document your own prompt refinements, failure
  cases, and evaluation as required by the assessment brief.


