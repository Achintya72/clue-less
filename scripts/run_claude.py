"""Run a frontier (or any LiteLLM) model against the MinuteCryptic env.

`mesocosm run local` only allows Ollama models, so this reuses Mesocosm's own
`bench()` engine directly (same scoring, traces, agent loop) with the platform
model allowlist skipped — letting us score Claude/Gemini/etc. locally.

The Anthropic key is read from the environment or from a gitignored `.env.local`
in the repo root (format: `ANTHROPIC_API_KEY=sk-...`).

Prereqs: `python adapter.py` running in another terminal (env on :8765).

Examples:
    python scripts/run_claude.py                       # claude sonnet, 10 episodes
    python scripts/run_claude.py --episodes 20
    python scripts/run_claude.py --model anthropic/claude-opus-4-8
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> None:
    """Minimal .env.local loader (no dependency) — only sets keys not already set."""
    f = ROOT / ".env.local"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seeds", default=None, help="e.g. 0-19 or 0,3,7 (overrides --episodes)")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--env-url", default="http://localhost:8765")
    ap.add_argument("--manifest", default=str(ROOT / "benchanything.json"))
    args = ap.parse_args()

    load_dotenv()
    if args.model.startswith("anthropic/") and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set. Put it in clue-less/.env.local or export it.")

    # Be resilient to provider rate limits: litellm retries 429s with backoff
    # (respects Retry-After), so low-RPM accounts complete instead of failing.
    import litellm
    litellm.num_retries = 10

    from bench_common.env_sdk.manifest import domain_config_from_manifest
    from bench_common.inference.bench import bench

    def parse_seeds(s):
        if not s:
            return None
        if "-" in s and "," not in s:
            lo, hi = s.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(x) for x in s.split(",")]

    domain = domain_config_from_manifest(args.manifest, env_url=args.env_url)
    result = asyncio.run(
        bench(
            model=args.model,
            domain_id=domain.id,
            env_url=args.env_url,
            num_episodes=args.episodes,
            seed_set=parse_seeds(args.seeds),
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            domain=domain,
            allow_any_model=True,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
