#!/bin/bash
# run.sh
#
# Launches both the Bokeh server (SINDy module backend) and the Flask app
# (frontend) with a single command, then automatically opens the browser
# once the servers are ready — no more manual copy-paste of the URL.
#
# Usage:
#   chmod +x run.sh   (only needed once)
#   ./run.sh
#
# Press Ctrl+C once to stop BOTH processes cleanly.

URL="http://127.0.0.1:8080"

echo "🚀 Starting Bokeh server..."
bokeh serve --allow-websocket-origin=127.0.0.1:8080 main.py &
BOKEH_PID=$!

echo "🚀 Starting Flask app..."
python3 flask_app.py &
FLASK_PID=$!

# When Ctrl+C is pressed, kill both background processes before exiting.
trap "echo '🛑 Stopping...'; kill -9 $BOKEH_PID $FLASK_PID; exit" SIGINT SIGTERM

# ── Auto-open browser ────────────────────────────────────────────────
# Wait a few seconds for Flask/Bokeh to actually start listening before
# opening the tab — opening too early would just show "connection refused".
sleep 3
echo "🌐 Opening $URL ..."
open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null
# `open` = macOS, `xdg-open` = Linux fallback (in case you switch OS later)

# Wait for both processes — script stays alive until either exits or Ctrl+C.
wait $BOKEH_PID $FLASK_PID