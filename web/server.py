import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from debate_app.config import Settings
from debate_app.engine import DebateEngine
from debate_app.errors import DebateError
from debate_app.schemas import DebateInput, DebateMessage


ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="观点竞技场", docs_url=None, redoc_url=None)


@dataclass
class DebateSession:
    queue: Queue = field(default_factory=Queue)
    paused: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def send(self, event_type: str, payload: object) -> None:
        self.queue.put({"type": event_type, "payload": payload})


sessions: dict[str, DebateSession] = {}
sessions_lock = threading.Lock()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/debates")
def create_debate(request: DebateInput) -> dict[str, str]:
    debate_id = str(uuid4())
    session = DebateSession()
    with sessions_lock:
        sessions[debate_id] = session

    def worker() -> None:
        try:
            settings = Settings.from_env()
            engine = DebateEngine(settings)
            result = engine.run(
                request,
                on_message=lambda message: session.send("message", message.model_dump(mode="json")),
                pause_event=session.paused,
                on_status=lambda status: session.send("status", {"status": status}),
            )
            session.send("judgment", result.judgment.model_dump(mode="json"))
        except (DebateError, ValidationError) as exc:
            session.send("error", {"message": str(exc)})
        except Exception:
            session.send("error", {"message": "发生未预期错误，辩论已安全停止。"})
        finally:
            session.send("done", {})

    session.thread = threading.Thread(target=worker, name=f"debate-{debate_id}", daemon=True)
    session.thread.start()
    return {"debate_id": debate_id}


@app.get("/api/debates/{debate_id}/stream")
def stream_debate(debate_id: str) -> StreamingResponse:
    session = sessions.get(debate_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Debate not found")

    def events():
        while True:
            try:
                event = session.queue.get(timeout=15)
            except Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"event: {event['type']}\ndata: {json.dumps(event['payload'], ensure_ascii=False)}\n\n"
            if event["type"] == "done":
                break

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/debates/{debate_id}/pause")
def pause_debate(debate_id: str) -> dict[str, str]:
    session = sessions.get(debate_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Debate not found")
    session.paused.set()
    session.send("status", {"status": "paused"})
    return {"status": "paused"}


@app.post("/api/debates/{debate_id}/resume")
def resume_debate(debate_id: str) -> dict[str, str]:
    session = sessions.get(debate_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Debate not found")
    session.paused.clear()
    session.send("status", {"status": "running"})
    return {"status": "running"}


def run() -> None:
    import uvicorn
    import os

    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    run()

