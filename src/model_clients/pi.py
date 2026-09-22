"""Non-interactive pi model calls for benchmark baselines."""

from __future__ import annotations

import subprocess


PI_MODELS = {
    "deepseek-flash": "deepseek/deepseek-flash",
    "grok-4.7": "wokey/grok-4.7",
}


def run_pi_model(prompt: str, model: str, *, timeout: int = 180) -> str:
    """Call one pi model with extensions, tools, and context discovery disabled."""
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
        model,
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
        raise RuntimeError(f"pi model call failed: {detail}")
    response = result.stdout.strip()
    if not response:
        raise RuntimeError("pi model returned an empty response")
    return response
