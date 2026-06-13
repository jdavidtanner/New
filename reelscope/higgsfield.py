"""Optional live Higgsfield API client.

By default ReelScope just generates prompts/shot-lists for you to paste into the
Higgsfield app — no key needed. Once you have a Higgsfield Creator plan and an
API key, set HIGGSFIELD_API_KEY (and optionally HIGGSFIELD_API_SECRET) and call
``generate_video()`` to kick off generation straight from a brief.

The endpoint/field names follow Higgsfield's public Cloud API shape and are kept
in one place so they're easy to update if the API changes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .brief import HiggsfieldBrief

_BASE = os.environ.get("HIGGSFIELD_API_BASE", "https://platform.higgsfield.ai/v1")


class HiggsfieldError(RuntimeError):
    pass


class HiggsfieldClient:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("HIGGSFIELD_API_KEY")
        self.api_secret = api_secret or os.environ.get("HIGGSFIELD_API_SECRET", "")
        if not self.api_key:
            raise HiggsfieldError(
                "No Higgsfield API key. Set HIGGSFIELD_API_KEY to use live generation. "
                "Until then, use the generated prompts in higgsfield_briefs.md by pasting "
                "them into the Higgsfield app."
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.api_secret:
            headers["hf-secret"] = self.api_secret
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{_BASE}{path}"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(), method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            raise HiggsfieldError(f"Higgsfield API error ({e.code}): {body[:300]}") from e
        except urllib.error.URLError as e:
            raise HiggsfieldError(f"Could not reach Higgsfield: {e}") from e

    def generate_video(self, brief: HiggsfieldBrief, **overrides: Any) -> dict[str, Any]:
        """Submit a generation job built from a brief. Returns the API response
        (typically a job id / status you can poll)."""
        payload: dict[str, Any] = {
            "prompt": brief.higgsfield_prompt,
            "aspect_ratio": brief.settings.get("aspect_ratio", "9:16"),
            "motion": brief.settings.get("motion", ""),
            "style": brief.settings.get("style", ""),
        }
        payload.update(overrides)
        return self._post("/text2video", payload)
