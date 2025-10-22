#!/usr/bin/env bash
# Smoke test for docker-compose stack.

set -euo pipefail

PROFILE=${PROFILE:-cpu}
API_URL=${API_URL:-http://localhost:8000}
JOB_PAYLOAD=${JOB_PAYLOAD:-'{"payload":{"source_uri":"","parameters":{}}}'}

cleanup() {
  docker compose down -v >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "[smoke] starting stack with profile=$PROFILE"
docker compose --profile "$PROFILE" up -d --build

echo "[smoke] waiting for API"
for _ in {1..30}; do
  if curl -fsS "$API_URL/v1/healthz" >/dev/null; then
    break
  fi
  sleep 1
done

JOB_ID=$(curl -fsS -X POST "$API_URL/v1/jobs" -H 'Content-Type: application/json' -d "$JOB_PAYLOAD" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "[smoke] submitted job $JOB_ID"
for _ in {1..60}; do
  STATUS_JSON=$(curl -fsS "$API_URL/v1/jobs/$JOB_ID")
  STATUS=$(printf '%s' "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  if [ "$STATUS" = "completed" ]; then
    ARTIFACT=$(printf '%s' "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['artifact_path'])")
    if [ -f "$ARTIFACT" ]; then
      echo "[smoke] artifact available at $ARTIFACT"
      exit 0
    fi
  fi
  sleep 1
done

echo "[smoke] job did not complete in time" >&2
exit 1
