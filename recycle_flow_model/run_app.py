import os
import sys
import traceback
import webbrowser


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == "__main__":
    print("Starting CHE 031 Evaporative Crystallization Application...")

    try:
        # Ensure working directory points to PyInstaller temporary extraction folder
        if hasattr(sys, "_MEIPASS"):
            print(f"Running inside PyInstaller bundle. Temporary dir: {sys._MEIPASS}")
            os.chdir(sys._MEIPASS)
        else:
            print("Running in standard Python environment.")

        from bokeh.command.bootstrap import main

        script_path = resource_path("combined_app.py")
        print(f"Target script path: {script_path}")

        # Verify target script existence
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Could not locate {script_path}")

        # Automatically open web browser to the default Bokeh app port
        print("Opening browser at http://localhost:5006/combined_app")
        webbrowser.open_new("http://localhost:5006/combined_app")

        # Construct argument list and pass explicitly to Bokeh's main bootstrap module
        args = ["bokeh", "serve", "--show", script_path]
        print("Launching Bokeh serve engine...")
        main(args)

    except Exception:
        print("\n" + "=" * 60)
        print("FATAL ERROR: Application crashed on startup!")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("\nPress ENTER to exit and close this window...")