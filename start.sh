#!/usr/bin/env sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BACKEND_MODE=${BACKEND_MODE:-local}

if [ "${1:-}" = "--cuda" ]; then
  BACKEND_MODE=cuda
fi

cleanup() {
  if [ "${BACKEND_MODE}" = "cuda" ]; then
    (
      cd "$ROOT_DIR"
      docker compose -f docker-compose.cuda.yml down >/dev/null 2>&1 || true
    )
  fi

  if [ -n "${BACKEND_PID:-}" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi

  if [ -n "${FRONTEND_PID:-}" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

if [ "${BACKEND_MODE}" = "cuda" ]; then
  echo 'Starting backend with CUDA Docker on http://localhost:8000'
else
  echo 'Starting backend on http://localhost:8000'
fi
(
  cd "$ROOT_DIR"
  if [ "${BACKEND_MODE}" = "cuda" ]; then
    docker compose -f docker-compose.cuda.yml up --build
  elif [ -x "$ROOT_DIR/.venv/bin/uvicorn" ]; then
    "$ROOT_DIR/.venv/bin/uvicorn" api.app:app --host 0.0.0.0 --port 8000 --reload
  elif command -v uv >/dev/null 2>&1; then
    uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
  elif command -v uvicorn >/dev/null 2>&1; then
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
  else
    echo 'Error: uvicorn not found. Run `uv sync` first.' >&2
    exit 1
  fi
) &
BACKEND_PID=$!

echo 'Starting frontend on http://localhost:3000'
(
  cd "$ROOT_DIR/frontend"
  pnpm dev
) &
FRONTEND_PID=$!

wait
