"""Small wrappers for reproducible non-interactive pi agent calls."""

from __future__ import annotations

import subprocess


PI_MODEL = "deepseek/deepseek-flash"


def run_pi_text(prompt: str, *, timeout: int = 120) -> str:
    """Return one print-mode response from pi without extensions or tools."""
    command = [
        "pi",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-tools",
        "--no-session",
        "--thinking",
        "off",
        "--model",
        PI_MODEL,
        "-p",
        "--",
        prompt,
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"pi generation failed: {detail}")
    response = result.stdout.strip()
    if not response:
        raise RuntimeError("pi generation returned an empty response")
    return response
