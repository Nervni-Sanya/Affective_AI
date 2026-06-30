#!/bin/bash
# SessionStart hook for Claude Code on the web.
# Installs the project's Python dependencies into the fresh web container so the
# test suite and the EmotiveLLM pipeline work from the first turn.
#
# Synchronous on purpose: torch is a heavy install and the whole test suite
# depends on it, so we want deps ready before the session starts (no races).
set -euo pipefail

# Web-only: no-op in local sessions, where deps are already managed.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Editable install pulls the core deps (torch, transformers, numpy, pyyaml) plus
# pytest from the [dev] extra, and makes the `emotive_llm` package importable.
# `install` (not `ci`) lets the cached container reuse the resolved environment.
python -m pip install -e ".[dev]"

echo "session-start: dependencies installed."
