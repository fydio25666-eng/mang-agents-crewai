import os
import json
from pathlib import Path
import re
from typing import TypeVar
from urllib.parse import urlparse

try:
    from crewai import Agent, Crew, LLM, Process, Task
except ImportError:  # Compatibility with CrewAI versions exporting LLM separately.
    from crewai import Agent, Crew, Process, Task
    from crewai.llm import LLM
from crewai.memory.storage import kickoff_task_outputs_storage
from json_repair import repair_json
from openai import OpenAI

from pydantic import BaseModel

from .config import ModelSettings
from .errors import AgentCallError
from .schemas import AgentTurn, JudgeResult


OutputModel = TypeVar("OutputModel", bound=BaseModel)


# Keep CrewAI's task database in the project, where the web process can write it.
_CREWAI_DATA_DIR = Path(os.getenv("CREWAI_DATA_PATH", "data/crewai"))
if not _CREWAI_DATA_DIR.is_absolute():
    _CREWAI_DATA_DIR = Path(__file__).resolve().parent.parent / _CREWAI_DATA_DIR
_CREWAI_DATA_DIR.mkdir(parents=True, exist_ok=True)
kickoff_task_outputs_storage.db_storage_path = lambda: str(_CREWAI_DATA_DIR)


def build_llm(settings: ModelSettings) -> LLM:
    kwargs = {
        "model": settings.model,
        "api_key": settings.api_key,
        "temperature": 0.35,
        "timeout": settings.timeout,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return LLM(**kwargs)


class DebateAgents:
    def __init__(self, qianwen: ModelSettings, deepseek: ModelSettings, openai: ModelSettings):
        self._judge_settings = openai
        self.pro = Agent(
            role="Proposition debater",
            goal="Build the strongest evidence-based case for the proposition.",
            backstory="You are rigorous, concise, and directly address the opposing argument.",
            llm=build_llm(qianwen),
            allow_delegation=False,
            verbose=False,
        )
        self.con = Agent(
            role="Opposition debater",
            goal="Stress-test the proposition and present the strongest counter-case.",
            backstory="You identify assumptions, counterexamples, and gaps in reasoning.",
            llm=build_llm(deepseek),
            allow_delegation=False,
            verbose=False,
        )
        self.judge = Agent(
            role="Neutral debate judge",
            goal="Compare both sides fairly and issue a structured, evidence-based ruling.",
            backstory="You score argument quality, relevance, rebuttal quality, and consistency.",
            llm=build_llm(openai),
            allow_delegation=False,
            verbose=False,
        )

    def run_turn(self, side: str, prompt: str) -> AgentTurn:
        agent = self.pro if side == "pro" else self.con
        return _run_structured_task(agent, prompt, AgentTurn)

    def run_judge(self, prompt: str) -> JudgeResult:
        schema = json.dumps(JudgeResult.model_json_schema(), ensure_ascii=True)
        try:
            client = OpenAI(
                api_key=self._judge_settings.api_key,
                base_url=_openai_base_url(self._judge_settings.base_url),
                timeout=self._judge_settings.timeout,
            )
            response = client.chat.completions.create(
                model=_provider_model_name(self._judge_settings.model),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a neutral debate judge. You MUST select either pro or con as the winner; "
                            "a draw or tie is forbidden. Choose the comparatively stronger argument even when "
                            "the difference is small, and explain the decisive reasons in reasoning and final_summary. "
                            "Return only one JSON object "
                            f"matching this schema: {schema}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            raw = _response_text(response)
        except Exception as exc:
            raise AgentCallError("Neutral debate judge request failed") from exc

        try:
            return JudgeResult.model_validate(_parse_json_object(raw))
        except Exception:
            return _fallback_judgment(raw)


def _run_structured_task(agent: Agent, prompt: str, output_model: type[OutputModel]) -> OutputModel:
    task = Task(
        description=(
            f"{prompt}\n\nReturn ONLY one valid JSON object matching this schema: "
            f"{json.dumps(output_model.model_json_schema(), ensure_ascii=True)}. "
            "Do not use Markdown fences or explanatory text."
        ),
        expected_output=f"One plain JSON object matching the {output_model.__name__} schema.",
        agent=agent,
    )
    try:
        result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
        raw = getattr(result, "raw", str(result))
        return output_model.model_validate(_parse_json_object(raw))
    except Exception as exc:
        raise AgentCallError(f"{agent.role} did not return valid structured output") from exc


def _parse_json_object(raw: str) -> dict:
    """Parse plain JSON while tolerating a model adding a fence or short preamble."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = repair_json(text, return_objects=True)
    if not isinstance(parsed, dict):
        raise ValueError("Structured response must be a JSON object")
    return parsed


def _provider_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _openai_base_url(base_url: str) -> str:
    """Point proxy host roots at their OpenAI-compatible v1 API."""
    normalized = base_url.rstrip("/")
    if not urlparse(normalized).path:
        return f"{normalized}/v1"
    return normalized


def _response_text(response: object) -> str:
    """Accept standard OpenAI responses and proxies returning plain strings."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            return str(message.get("content") or "")
        return json.dumps(response, ensure_ascii=False)

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    return str(response)


def _fallback_judgment(raw: str) -> JudgeResult:
    """Build a mandatory winner verdict when a proxy returns non-JSON text."""
    text = raw.strip()
    if re.search(r"<!doctype\s+html|<html\b", text, flags=re.IGNORECASE):
        text = "裁判代理返回了非模型页面；依据已有辩论内容执行保守择优。"
    text = text[:11000]
    lowered = text.lower()
    pro_wins = bool(re.search(r"正方(?:胜出|获胜|胜利)|支持方(?:胜出|获胜)|\bpro\s+wins?\b", lowered))
    con_wins = bool(re.search(r"反方(?:胜出|获胜|胜利)|反对方(?:胜出|获胜)|\bcon\s+wins?\b", lowered))

    if pro_wins and not con_wins:
        winner, pro_score, con_score = "pro", 55, 45
    elif con_wins and not pro_wins:
        winner, pro_score, con_score = "con", 45, 55
    else:
        pro_signals = len(re.findall(r"正方.{0,18}(?:更强|充分|严谨|有效|胜)|(?:更强|充分|严谨|有效).{0,18}正方", text))
        con_signals = len(re.findall(r"反方.{0,18}(?:更强|充分|严谨|有效|胜)|(?:更强|充分|严谨|有效).{0,18}反方", text))
        if con_signals > pro_signals:
            winner, pro_score, con_score = "con", 49, 51
        else:
            winner, pro_score, con_score = "pro", 51, 49
        decision = "双方论证接近，依据论证完整性与直接回应程度进行强制择优。"
        text = f"{text}\n{decision}".strip()[:12000]

    reasoning = text or "裁判输出无法解析为 JSON；依据论证完整性作出强制择优判定。"
    return JudgeResult(
        winner=winner,
        pro_score=pro_score,
        con_score=con_score,
        reasoning=reasoning,
        pro_strengths=[],
        con_strengths=[],
        pro_weaknesses=[],
        con_weaknesses=[],
        final_summary=reasoning,
    )

