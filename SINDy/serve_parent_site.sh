#!/usr/bin/env bash
set -euo pipefail

SINDY_PORT="${SINDY_PORT:-5006}"
SINDY_ADDRESS="${SINDY_ADDRESS:-127.0.0.1}"
SINDY_WEBSOCKET_ORIGIN="${SINDY_WEBSOCKET_ORIGIN:-srrweb.cc.lehigh.edu}"
SINDY_URL_PREFIX="${SINDY_URL_PREFIX:-/sindy-bokeh}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
exec bokeh serve main.py \
  --port "$SINDY_PORT" \
  --address "$SINDY_ADDRESS" \
  --prefix "$SINDY_URL_PREFIX" \
  --allow-websocket-origin "$SINDY_WEBSOCKET_ORIGIN" \
  --use-xheaders
