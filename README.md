# CrewAI Debate

This project runs a bounded two-sided debate and asks a third model to judge it.

## Components

- Pro side: QIANWEN from `.env`
- Con side: DEEPSEEK from `.env`
- Judge: OPENAI from `.env`
- Persistence: local SQLite database
- Maximum: 12 rounds and 24 debate messages
- Output: validated Pydantic models serialized as JSON

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

Fill the three API keys in `.env`. The existing `.env` is never tracked by Git.

## Run

Start the HTML web interface:

```powershell
python -m web.server
```

Then open `http://127.0.0.1:8000`. The page receives each completed turn through an SSE stream. The pause button takes effect between model calls and resume continues the same debate.

Or run a debate from the command line:

```powershell
python -m debate_app.main --topic "Remote work" --question "Should a company become remote-first?" --rounds 6
```

The default database is `data/debates.sqlite3`. Each message is persisted after its turn, so an interrupted debate keeps its completed transcript.

## Deploy To Render

This repository includes `render.yaml`. Create a Blueprint from the repository in Render and set the three `*_API_KEY` secrets when prompted. The Blueprint uses one web worker and a 1 GB persistent disk mounted at `/var/data`, so SQLite history survives restarts. The `starter` plan is required for a persistent disk; a free instance can run without a disk but its local history is ephemeral.

Equivalent manual settings:

```text
Build Command: pip install .
Start Command: uvicorn web.server:app --host 0.0.0.0 --port $PORT --workers 1
Health Check Path: /health
```

Do not commit `.env`, API keys, `data/`, or SQLite files. Configure secrets in Render Environment Variables.

## Configuration

The application accepts `QIANWEN_API_KEY`, `DEEPSEEK_API_KEY`, and `OPENAI_API_KEY`. It also accepts `QIANWEN`, `DEEPSEEK`, and `OPENAI` as legacy key names. Model names and OpenAI-compatible base URLs can be overridden with the corresponding `*_MODEL` and `*_BASE_URL` variables.

