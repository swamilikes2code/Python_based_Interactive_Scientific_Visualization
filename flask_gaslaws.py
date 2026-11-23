import atexit
import subprocess
import sys
from pathlib import Path

from bokeh.embed import server_document
from flask import Flask, abort, render_template_string, send_from_directory

BASE = Path(__file__).parent
BOKEH_FILE = "Gas_Laws.py"          # your Bokeh app file
BOKEH_NAME = Path(BOKEH_FILE).stem  # becomes the URL path, same case
HTML_FILE = "gas_laws_page1.html"    # your styled page

# Start the Bokeh server automatically
bokeh_proc = subprocess.Popen([
    sys.executable, "-m", "bokeh", "serve", str(BASE / BOKEH_FILE),
    "--port", "5006",
    "--allow-websocket-origin=localhost:5006",
    "--allow-websocket-origin=localhost:8080",
])
atexit.register(bokeh_proc.terminate)

app = Flask(__name__)


@app.route("/")
def index():
    html = (BASE / HTML_FILE).read_text(encoding="utf-8")
    script = server_document(f"http://localhost:5006/{BOKEH_NAME}")
    return render_template_string(html, script=script)


@app.route("/<path:filename>")
def images(filename):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return send_from_directory(BASE, filename)
    abort(404)


if __name__ == "__main__":
    app.run(port=8080, debug=False)