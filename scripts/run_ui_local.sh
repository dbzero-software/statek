#!/bin/bash
# Start the Statek Dashboard web UI locally.
# Environment is loaded from .env_statek in the project root.
#
# Usage:
#   ./scripts/run_ui_local.sh          - start the web UI
#   ./scripts/run_ui_local.sh stop     - stop a running instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE=/tmp/statek_webui.log
ENV_FILE="$PROJECT_ROOT/.env_statek"

stop_service() {
    echo "Stopping Statek Dashboard..."
    pkill -f "python3 -m StatekWebUI.main" 2>/dev/null || true
    echo "Stopped."
}

if [ "$1" = "stop" ]; then
    stop_service
    exit 0
fi

stop_service
sleep 1

cd "$PROJECT_ROOT"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found." >&2
    exit 1
fi

set -a; . "$ENV_FILE"; set +a

DATA_PREFIX="/${STATEK_ORG_NAME}/${STATEK_PROJECT_NAME}/${STATEK_ENV}/data"

# Build --open-prefix flags from STATEK_OPEN_PREFIXES (colon-separated) + DATA_PREFIX
OPEN_PREFIX_ARGS="--open-prefix ${DATA_PREFIX}"
if [ -n "${STATEK_OPEN_PREFIXES:-}" ]; then
    IFS=':' read -ra _PREFIXES <<< "$STATEK_OPEN_PREFIXES"
    for _p in "${_PREFIXES[@]}"; do
        [ -n "$_p" ] && OPEN_PREFIX_ARGS="$OPEN_PREFIX_ARGS --open-prefix $_p"
    done
fi

echo "Starting Statek Dashboard..."
python3 -m StatekWebUI.main \
    --host "${STATEK_UI_HOST:-0.0.0.0}" \
    --port "${STATEK_UI_PORT:-8765}" \
    --db0-path /selltime_data/ \
    --import selltime.ai.statek_root \
    --import selltime.ai.selltime_coordinator \
    --import selltime.ai.selltime_dispatcher \
    $OPEN_PREFIX_ARGS \
    > "$LOG_FILE" 2>&1 &

WEBUI_PID=$!
echo "  PID: $WEBUI_PID  (log: $LOG_FILE)"
echo "  URL: http://localhost:${STATEK_UI_PORT:-8888}"
echo ""
echo "To stop: ./scripts/run_ui_local.sh stop"
