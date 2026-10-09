import argparse
import sys

from .config import Settings
from .engine import DebateEngine
from .errors import DebateError
from .schemas import DebateInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded CrewAI debate.")
    parser.add_argument("--topic", required=True, help="Debate topic")
    parser.add_argument("--question", required=True, help="Question to debate")
    parser.add_argument("--context", default="", help="Optional context")
    parser.add_argument("--rounds", type=int, default=12, help="Number of rounds, capped at 12")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        settings = Settings.from_env()
        request = DebateInput(
            topic=args.topic,
            question=args.question,
            context=args.context,
            max_rounds=min(args.rounds, 12),
        )
        result = DebateEngine(settings).run(request)
        print(result.model_dump_json(indent=2))
        return 0
    except DebateError as exc:
        print(f"Debate failed safely: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Debate interrupted; completed messages remain in local memory.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

