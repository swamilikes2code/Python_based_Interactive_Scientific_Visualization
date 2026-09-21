from flask import Flask, render_template
from bokeh.embed import server_document
import os

app = Flask(__name__)

# ── Deployment note ──────────────────────────────────────────────────────
# Locally, BOKEH_URL defaults to localhost:5006 (unchanged from before).
# On Render, this MUST be set as an environment variable pointing to the
# public URL of the deployed Bokeh service (e.g.
# https://sindy-bokeh.onrender.com/main), since the browser connects to
# this URL DIRECTLY via WebSocket — it does not go through Flask.
BOKEH_URL = os.environ.get("BOKEH_URL", "http://localhost:5006/main")


@app.route("/")
def index():
    script = server_document(BOKEH_URL)
    return render_template(
        "index.html",
        script=script
    )


@app.route("/fragment/storyline")
def storyline():
    return render_template("fragment/storyline.html")


@app.route("/fragment/about")
def about():
    return render_template("fragment/about.html")


@app.route("/fragment/training")
def train():
    return render_template("fragment/training.html")


@app.route("/fragment/testing")
def test():
    return render_template("fragment/testing.html")

@app.route("/fragment/predicting")
def predict():
    return render_template("fragment/predicting.html")


@app.route("/fragment/questions")
def questions():
    return render_template("fragment/questions.html")


if __name__ == "__main__":
    # ── Deployment note ──────────────────────────────────────────────────
    # Render injects a PORT environment variable and expects the service
    # to bind to it on 0.0.0.0 (not localhost/127.0.0.1, since Render's
    # load balancer connects from outside the container). Locally, this
    # falls back to port 8080 exactly as before — no change to local dev.
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)