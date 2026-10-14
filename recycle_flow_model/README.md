# CHE 031 Evaporative Crystallization

This directory contains Faith's Bokeh module and its standalone desktop
packaging. The macOS application runs a local Bokeh server on an available
port, then opens the module in the user's default browser.

## Run from source

```bash
python3 -m pip install -r requirements-build.txt
python3 run_app.py
```

## Build the macOS application

PyInstaller is not a cross-compiler. Build this artifact on macOS, preferably
on the oldest macOS version the application must support.

```bash
chmod +x build_macos.sh
./build_macos.sh
```

The result is `dist/CHE031 Crystallization.app`. An optional architecture may
be supplied:

```bash
./build_macos.sh arm64
./build_macos.sh x86_64
./build_macos.sh universal2
```

The selected Python interpreter and every binary dependency must support the
requested architecture. In practice, use an Apple Silicon Python environment
for `arm64`, an Intel Python environment for `x86_64`, or universal2 Python and
dependencies for `universal2`.

## Test checklist

1. Double-click the `.app` in Finder.
2. Confirm exactly one browser tab opens and the interactive module loads.
3. Switch between **No Recycle** and **With Recycle**.
4. Move the fresh-feed slider and verify the image, summary, and plot update.
5. Launch a second copy and verify both instances load without a port conflict.

Startup diagnostics are written to
`~/Library/Logs/CHE031_Crystallization_App.log`.

## Distribution note

Local builds receive ad-hoc signing only. For distribution outside the build
machine, the deployer should apply an Apple Developer ID signature and notarize
the application, or provide testers with the institution's approved Gatekeeper
instructions.
