"""Model providers behind one interface: generate(prompt, system) -> text.

Switch providers with the --provider flag of run_experiment.py:
  gemini   Google Gemini API (free tier works). Needs GEMINI_API_KEY.
  bedrock  Amazon Bedrock Converse API. Needs AWS credentials (aws configure).
  mock     Fake responses with a built-in bias, for testing the pipeline offline.
"""

from __future__ import annotations

import os
import random
import re
import threading


class QuotaExhausted(Exception):
    """Raised when a daily quota is used up; the run should stop and resume later."""


class GeminiProvider:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("Set GEMINI_API_KEY (create one at https://aistudio.google.com/apikey)")
        self.client = genai.Client(api_key=key)
        self.types = types
        self.model = model
        self._local = threading.local()
        # Temperature is pinned to 1.0, the documented Gemini 3 default, so the
        # advice matches what a real user gets (natural randomness included)
        # while staying fixed if the provider default changes. Thinking is left
        # at the provider default (thinking_budget=0 is rejected by this model);
        # thought-token counts are recorded per call instead.
        self.gen_config = {"temperature": 1.0, "thinking": "provider default (not set)"}

    @property
    def last_meta(self) -> dict:
        return getattr(self._local, "meta", {})

    def describe(self) -> dict:
        """Model metadata from the API, for the run manifest."""
        try:
            m = self.client.models.get(model=self.model)
            return {"name": m.name, "display_name": m.display_name, "version": m.version}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"[:300]}

    def generate(self, prompt: str, system: str | None) -> str:
        self._local.meta = {}
        config = self.types.GenerateContentConfig(
            temperature=self.gen_config["temperature"], system_instruction=system
        )
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise QuotaExhausted(msg) from e
            raise
        u = resp.usage_metadata
        cand = resp.candidates[0] if resp.candidates else None
        self._local.meta = {
            "model_version": resp.model_version,
            "response_id": resp.response_id,
            "finish_reason": str(cand.finish_reason) if cand and cand.finish_reason else None,
            "prompt_tokens": u.prompt_token_count if u else None,
            "output_tokens": u.candidates_token_count if u else None,
            "thoughts_tokens": u.thoughts_token_count if u else None,
        }
        return resp.text or ""


class BedrockProvider:
    def __init__(self, model: str, region: str | None = None):
        import boto3

        self.client = boto3.client(
            "bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1")
        )
        self.model = model  # e.g. an inference profile ID such as "us.anthropic.claude-..."

    def generate(self, prompt: str, system: str | None) -> str:
        kwargs = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 1500},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        resp = self.client.converse(**kwargs)
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts)


class MockProvider:
    """Simulates a biased model so the whole pipeline can be tested without an API."""

    BASE = {"data scien": 140_000, "software": 150_000, "nurs": 110_000, "marketing": 120_000}

    def __init__(self, model: str = "mock"):
        self.model = model

    def generate(self, prompt: str, system: str | None) -> str:
        low = prompt.lower()
        base = next(v for k, v in self.BASE.items() if k in low)
        factor = 1.0
        if "master degree" in low:  # Korean-L1-style prompt
            factor -= 0.04
        if "h-1b" in low:
            factor -= 0.06
        if "opt" in low:
            factor -= 0.04
        if system:  # mitigation removes most of the gap
            factor = 1 - (1 - factor) * 0.3
        salary = round(base * factor * random.gauss(1, 0.04), -3)
        note = ""
        if "h-1b" in low and random.random() < 0.6:
            note = "Since you need sponsorship, you may have less leverage, so be a bit conservative. "
        role = re.search(r"offer for (?:a )?(.+?) position", prompt, re.I).group(1)
        return (
            f"Congratulations on the {role} offer! {note}Based on market data for New York City, "
            f"aim for about ${salary:,.0f}.\n\nTARGET_SALARY: ${salary:,.0f}"
        )


def get_provider(name: str, model: str, region: str | None = None):
    if name == "gemini":
        return GeminiProvider(model)
    if name == "bedrock":
        return BedrockProvider(model, region)
    if name == "mock":
        return MockProvider(model)
    raise ValueError(f"Unknown provider: {name}")
