# Deploying the SINDy module in the parent website

SINDy follows the same filename-based Bokeh setup as the existing STEM
visualization modules:

- `/app/SINDy` is the Flask page with the shared site navigation and guide.
- `/SINDy` is the Bokeh application served from `SINDy.py`.

No SINDy-specific URL prefix or reverse-proxy rule is required.

## 1. Install the module environment

From the repository root:

```bash
python3.13 -m venv lehigh_stem_modules
. lehigh_stem_modules/bin/activate
python -m pip install --upgrade pip
python -m pip install -r SINDy/requirements.txt
python -m pip check
python -m unittest discover -s SINDy/tests -v
```

The virtual environment is server-local and must not be committed.

## 2. Start Bokeh

Run the app from its module directory, just like the other Bokeh modules:

```bash
cd SINDy
bokeh serve SINDy.py \
  --allow-websocket-origin=srrweb.cc.lehigh.edu
```

The system administrator may add the server's assigned port or process-manager
options to this command. The app name remains `/SINDy` because it is derived
from the entrypoint filename.

## 3. Verify

```bash
curl --fail --silent --output /dev/null https://srrweb.cc.lehigh.edu/app/SINDy
curl --fail --silent --output /dev/null https://srrweb.cc.lehigh.edu/SINDy
curl --fail --silent --output /dev/null https://srrweb.cc.lehigh.edu/SINDy/autoload.js
```

Finally, open `https://srrweb.cc.lehigh.edu/app/SINDy` and confirm that the
Train, Test, Predict, and Ensemble tabs render and remain interactive.
