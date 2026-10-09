import re

from .errors import SafetyError
from .schemas import AgentTurn, DebateInput, JudgeResult


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION_MARKERS = re.compile(
    r"(?:ignore\s+(?:all|any|previous)\s+instructions|reveal\s+(?:the\s+)?system\s+prompt|execute\s+shell)",
    re.IGNORECASE,
)


def validate_debate_input(data: DebateInput) -> DebateInput:
    text = f"{data.topic}\n{data.question}\n{data.context}"
    if _CONTROL_CHARS.search(text):
        raise SafetyError("Input contains unsupported control characters")
    if _INJECTION_MARKERS.search(text):
        raise SafetyError("Input contains an instruction-injection pattern")
    return data


def validate_agent_turn(turn: AgentTurn, max_chars: int) -> AgentTurn:
    if len(turn.content) > max_chars:
        raise SafetyError("Agent output exceeds the configured size limit")
    if _CONTROL_CHARS.search(turn.content):
        raise SafetyError("Agent output contains unsupported control characters")
    return turn


def validate_judgment(judgment: JudgeResult, max_chars: int) -> JudgeResult:
    if len(judgment.reasoning) > max_chars or len(judgment.final_summary) > max_chars:
        raise SafetyError("Judge output exceeds the configured size limit")
    return judgment

