#!/usr/bin/env python3
"""Deploy the maintained NL2Hull demo files to a Hugging Face Gradio Space."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = PROJECT_ROOT / "src" / "demo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="wenhuahuo/nl2hull-demo",
        help="Hugging Face Space repository ID",
    )
    args = parser.parse_args()

    required = [
        DEMO_ROOT / "app.py",
        DEMO_ROOT / "requirements.txt",
        DEMO_ROOT / "README.md",
        DEMO_ROOT / "assets" / "KVLCC2.json",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing demo files: {', '.join(missing)}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="cpu-basic",
        private=False,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="nl2hull_space_upload_") as temporary:
        staging = Path(temporary)
        shutil.copy2(DEMO_ROOT / "app.py", staging / "app.py")
        shutil.copy2(DEMO_ROOT / "requirements.txt", staging / "requirements.txt")
        shutil.copy2(DEMO_ROOT / "README.md", staging / "README.md")
        (staging / "assets").mkdir()
        shutil.copy2(DEMO_ROOT / "assets" / "KVLCC2.json", staging / "assets" / "KVLCC2.json")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=staging,
            commit_message="feat: deploy NL2Hull Gradio demo",
        )

    print(f"deployed https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
