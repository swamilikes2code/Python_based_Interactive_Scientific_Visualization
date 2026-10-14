# CHE 031 Gas Laws

This directory contains Erin's Gas Laws Bokeh module and standalone desktop
packaging. The macOS application starts a local Bokeh server on an available
port and opens the module in the user's default browser.

## Run from source

```bash
python3 -m pip install -r requirements-build.txt
python3 run_app.py
```

## Build the macOS application

PyInstaller is not a cross-compiler. Build on macOS, preferably on the oldest
macOS version that must be supported.

```bash
chmod +x build_macos.sh
./build_macos.sh
```

The result is `dist/CHE031 Gas Laws.app`. An optional architecture may be
supplied:

```bash
./build_macos.sh arm64
./build_macos.sh x86_64
./build_macos.sh universal2
```

The Python interpreter and binary dependencies must support the selected
architecture.

## Test checklist

1. Double-click the `.app` in Finder.
2. Confirm exactly one browser tab opens and the Gas Laws plot loads.
3. Select every substance, including **Carbon Dioxide**.
4. Switch among temperature, volume, and pressure comparisons.
5. Move each available slider and verify real-gas CSV data and model curves
   update.
6. Launch a second copy and verify both instances load without a port conflict.

Startup diagnostics are written to `~/Library/Logs/CHE031_Gas_Laws.log`.

## Distribution note

Local builds receive ad-hoc signing only. For distribution outside the build
machine, the deployer should apply an Apple Developer ID signature and notarize
the application, or provide testers with the institution's approved Gatekeeper
instructions.
