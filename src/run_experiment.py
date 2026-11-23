"""Run the audit: send every condition to a model, `reps` times, and save raw responses.

Results are appended to data/raw/<provider>__<model>.jsonl, one line per call.
The run is resumable: re-running the same command skips calls already saved,
so a free-tier daily quota can be finished over more than one session.

Calls are made in "waves": every condition gets repetition 0 (in random order),
then every condition gets repetition 1, and so on. If a run stops early, the
data collected so far is still balanced across conditions.

Examples
  python src/run_experiment.py --provider mock --model mock --reps 3
  python src/run_experiment.py --provider gemini --model gemini-flash-latest --reps 5 --rpm 8
  python src/run_experiment.py --provider bedrock --model us.anthropic.claude-haiku-4-5-20251001-v1:0 --reps 10 --rpm 40 --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompts import all_conditions  # noqa: E402
from providers import QuotaExhausted, get_provider  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class RateLimiter:
    """Spaces calls evenly so we stay under a requests-per-minute limit."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            wait_for = self.next_time - now
            self.next_time = max(now, self.next_time) + self.interval
        if wait_for > 0:
            time.sleep(wait_for)


def call_with_retry(provider, limiter, prompt, system, max_tries=5):
    for attempt in range(max_tries):
        limiter.wait()
        try:
            start = time.monotonic()
            text = provider.generate(prompt, system)
            return text, time.monotonic() - start, None
        except QuotaExhausted:
            raise
        except Exception as e:  # noqa: BLE001  (rate limits, timeouts, transient errors)
            if attempt == max_tries - 1:
                return None, None, f"{type(e).__name__}: {e}"[:500]
            time.sleep(min(60, 5 * 2 ** attempt))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def write_manifest(out_path: Path, args, provider, gen_config, n_tasks: int):
    """Append this invocation's settings to <out_file>.manifest.json (one entry per run)."""
    src = Path(__file__).parent
    run = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "provider": args.provider,
        "model": args.model,
        "model_info": provider.describe() if hasattr(provider, "describe") else None,
        "reps": args.reps,
        "seed": args.seed,
        "rpm": args.rpm,
        "workers": args.workers,
        "no_mitigation": args.no_mitigation,
        "limit": args.limit,
        "calls_planned": n_tasks,
        "gen_config": gen_config,
        "git_commit": _git_commit(),
        "file_sha256": {f: _sha256(src / f) for f in ("prompts.py", "run_experiment.py", "providers.py")},
        "python": platform.python_version(),
        "sdk_versions": {
            pkg: _pkg_version(pkg) for pkg in ("google-genai", "boto3", "botocore")
        },
    }
    mpath = out_path.with_suffix(".manifest.json")
    data = json.loads(mpath.read_text()) if mpath.exists() else {"runs": []}
    data["runs"].append(run)
    mpath.write_text(json.dumps(data, indent=2))


def _pkg_version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=["gemini", "bedrock", "mock"])
    ap.add_argument("--model", required=True, help="Model ID exactly as the provider lists it")
    ap.add_argument("--reps", type=int, default=10, help="Repetitions per condition (default 10)")
    ap.add_argument("--rpm", type=float, default=8, help="Max requests per minute (default 8, safe for free tiers)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (keep 1 on free tiers)")
    ap.add_argument("--region", default=None, help="AWS region for Bedrock")
    ap.add_argument("--no-mitigation", action="store_true", help="Skip the fairness-system-prompt arm (halves the calls)")
    ap.add_argument("--limit", type=int, default=None, help="Stop after this many new calls (for a quick pilot)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mitigations = ("none",) if args.no_mitigation else ("none", "fairness")
    conditions = all_conditions(mitigations)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = RAW_DIR / f"{args.provider}__{safe_model}.jsonl"

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("error") is None:
                done.add((rec["cond_id"], rec["rep"]))

    rng = random.Random(args.seed)
    tasks = []
    for rep in range(args.reps):
        wave = [c for c in conditions if (c["cond_id"], rep) not in done]
        rng.shuffle(wave)
        tasks.extend((c, rep) for c in wave)
    if args.limit:
        tasks = tasks[: args.limit]

    total = len(conditions) * args.reps
    print(f"{len(conditions)} conditions x {args.reps} reps = {total} calls; "
          f"{len(done)} already done, {len(tasks)} to run -> {out_path}")
    if not tasks:
        return

    provider = get_provider(args.provider, args.model, args.region)
    gen_config = getattr(provider, "gen_config", None)
    write_manifest(out_path, args, provider, gen_config, n_tasks=len(tasks))
    limiter = RateLimiter(args.rpm)
    write_lock = threading.Lock()
    stop = threading.Event()
    counts = {"ok": 0, "err": 0}

    def run_one(cond, rep):
        if stop.is_set():
            return
        try:
            text, latency, err = call_with_retry(provider, limiter, cond["prompt"], cond["system"])
        except QuotaExhausted as e:
            stop.set()
            print(f"\nDaily quota used up: {str(e)[:200]}\n"
                  "Progress is saved. Re-run the same command after the quota resets "
                  "(Gemini resets at midnight Pacific time).")
            return
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": args.provider,
            "model": args.model,
            "rep": rep,
            **{k: cond[k] for k in ("cond_id", "occupation", "style", "visa", "mitigation", "prompt", "system")},
            "response": text,
            "latency_s": round(latency, 2) if latency else None,
            "error": err,
            "seed": args.seed,
            "gen_config": gen_config,
            # Same thread as the call, so this is this call's metadata.
            **(getattr(provider, "last_meta", {}) if not err else {}),
        }
        with write_lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            counts["err" if err else "ok"] += 1
            n = counts["ok"] + counts["err"]
            if n % 10 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} done ({counts['err']} errors)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, c, r) for c, r in tasks]
        for _ in as_completed(futures):
            pass

    print(f"Finished: {counts['ok']} ok, {counts['err']} errors. "
          "Errors are not counted as done, so re-running retries them.")


if __name__ == "__main__":
    main()
