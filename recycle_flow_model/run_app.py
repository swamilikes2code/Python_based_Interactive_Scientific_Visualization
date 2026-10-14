"""Desktop launcher for the CHE 031 crystallization Bokeh application."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from bokeh.application import Application
from bokeh.application.handlers import ScriptHandler
from bokeh.server.server import Server


APP_ROUTE = "/combined_app"
APP_NAME = "CHE031_Crystallization_App"


def resource_path(filename: str) -> Path:
    """Return a resource path in source and PyInstaller builds."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / filename


def configure_logging() -> Path:
    """Write startup errors somewhere a tester can attach to a bug report."""
    if sys.platform == "darwin":
        log_dir = Path.home() / "Library" / "Logs"
    else:
        log_dir = Path.home() / f".{APP_NAME}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{APP_NAME}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return log_path


def main() -> None:
    log_path = configure_logging()
    script_path = resource_path("combined_app.py")
    if not script_path.is_file():
        raise FileNotFoundError(f"Could not locate bundled app: {script_path}")

    logging.info("Starting application from %s", script_path)
    application = Application(ScriptHandler(filename=str(script_path)))

    # Port 0 asks the operating system for a free port, avoiding collisions
    # with another local Bokeh application already using port 5006.
    server = Server({APP_ROUTE: application}, port=0)
    server.start()
    logging.info("Bokeh server listening on port %s", server.port)

    # Open the browser only after the server is ready.
    server.io_loop.add_callback(server.show, APP_ROUTE)
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception:
        logging.exception("Application crashed; log file: %s", log_path)
        raise
    finally:
        server.stop()


if __name__ == "__main__":
    main()
