#!/usr/bin/env python3
"""Secure GitHub clone/update helper for Google Colab.

The helper reads a token from Colab Secrets only when needed and passes it to
Git through a temporary GIT_ASKPASS process. The token is never written to the
repository, remote URL, notebook output, or Google Drive.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/chenzizi07/KMFM.git"
SECRET_NAMES = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_PAT",
    "SASM_MAMBA_GITHUB_TOKEN",
    "SASM_MAMBA_PAT",
)


def _colab_token() -> str | None:
    try:
        from google.colab import userdata
    except ImportError:
        return None
    for name in SECRET_NAMES:
        try:
            value = userdata.get(name)
        except Exception:  # Secret lookup errors should fall back to public Git.
            continue
        if value:
            return str(value)
    return None


def _run_git(args: list[str], cwd: Path | None, token: str | None) -> None:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
    askpass_path: Path | None = None
    if token:
        askpass_dir = Path(tempfile.mkdtemp(prefix="kmfm_git_askpass_"))
        askpass_path = askpass_dir / "askpass.sh"
        askpass_path.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*|*username*) printf '%s\\n' 'x-access-token' ;;\n"
            "  *) printf '%s\\n' \"$KMFM_GIT_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass_path.chmod(0o700)
        env["GIT_ASKPASS"] = str(askpass_path)
        env["KMFM_GIT_TOKEN"] = token
    try:
        subprocess.run(["git", *args], cwd=cwd, env=env, check=True)
    finally:
        if askpass_path is not None:
            shutil.rmtree(askpass_path.parent, ignore_errors=True)


def _install_dependencies(project_dir: Path) -> None:
    requirements = project_dir / "requirements-colab.txt"
    if requirements.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(project_dir)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone or fast-forward-update KMFM in Colab")
    parser.add_argument("--repo-url", default=os.environ.get("KMFM_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--mode", choices=("clone", "update"), default="update")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    project_dir.parent.mkdir(parents=True, exist_ok=True)
    token = _colab_token()

    if args.mode == "clone":
        if project_dir.exists():
            if (project_dir / ".git").exists():
                args.mode = "update"
            elif any(project_dir.iterdir()):
                raise SystemExit(
                    f"Refusing to overwrite non-git directory: {project_dir}. "
                    "Move it aside or choose a new project path."
                )
            else:
                project_dir.rmdir()
        if args.mode == "clone":
            _run_git(["clone", "--", args.repo_url, str(project_dir)], cwd=project_dir.parent, token=token)

    if not (project_dir / ".git").exists():
        raise SystemExit(
            f"{project_dir} is not a Git repository. Run this helper with --mode clone first."
        )
    _run_git(["pull", "--ff-only"], cwd=project_dir, token=token)
    for folder in ("results", "reports", "splits"):
        (project_dir / folder).mkdir(parents=True, exist_ok=True)
    if not args.skip_install:
        _install_dependencies(project_dir)
    print(f"KMFM repository updated: {project_dir}")


if __name__ == "__main__":
    main()
