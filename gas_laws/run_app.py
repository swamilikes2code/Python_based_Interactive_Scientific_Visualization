import os
import sys
import traceback
import webbrowser

print("Starting Gas Laws application wrapper...")

try:
    # Ensure working directory points to PyInstaller temporary extraction folder
    if hasattr(sys, "_MEIPASS"):
        print(f"Running inside PyInstaller bundle. Temporary dir: {sys._MEIPASS}")
        os.chdir(sys._MEIPASS)
    else:
        print("Running in standard Python environment.")

    from bokeh.command.bootstrap import main

    def resource_path(relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    script_path = resource_path("Gas_Laws.py")
    print(f"Target script path: {script_path}")

    # Verify target script existence
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Could not locate {script_path}")

    # Automatically open web browser
    print("Opening browser at http://localhost:5006/Gas_Laws")
    webbrowser.open_new("http://localhost:5006/Gas_Laws")

    # Construct the argument list for Bokeh serve
    args = ["bokeh", "serve", "--show", script_path]

    # Pass args directly to main()
    print("Launching Bokeh serve engine...")
    main(args)

except Exception as e:
    print("\n" + "=" * 60)
    print("FATAL ERROR: Application crashed on startup!")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)
    input("\nPress ENTER to exit and close this window...")