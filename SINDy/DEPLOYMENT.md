# Deploying the SINDy module in the parent website

The integration uses two routes on the same public host:

- `/SINDy` is the Flask page containing the shared site navigation and guide.
- `/sindy-bokeh/main` is the Bokeh application embedded by that page.

Keeping these routes separate is required. Pointing both Flask and Bokeh at
`/SINDy` works on different local ports but conflicts behind one production
hostname.

## 1. Install the module environment

From the repository root:

```bash
python3 -m venv .venv-sindy
. .venv-sindy/bin/activate
python -m pip install -r SINDy/requirements.txt
```

The virtual environment is server-local and must not be committed.

## 2. Start Bokeh

```bash
./SINDy/serve_parent_site.sh
```

The launcher defaults are:

```text
SINDY_PORT=5006
SINDY_ADDRESS=127.0.0.1
SINDY_URL_PREFIX=/sindy-bokeh
SINDY_WEBSOCKET_ORIGIN=srrweb.cc.lehigh.edu
```

Each value can be overridden through the environment. Run this command under
the same process manager used for the other Bokeh modules so it restarts after
server reboots or process failures.

## 3. Configure the reverse proxy

Proxy the complete public prefix below to `127.0.0.1:5006`, preserving the
prefix and supporting WebSocket upgrades:

```text
/sindy-bokeh/  ->  http://127.0.0.1:5006/sindy-bokeh/
```

The WebSocket endpoint is below `/sindy-bokeh/main/ws`. Forward the original host,
protocol, and client IP headers. Leave `/SINDy` routed to the Flask service.

If a different public prefix is required, set matching values in both places:

```bash
SINDY_URL_PREFIX=/chosen-prefix ./SINDy/serve_parent_site.sh
export SINDY_BOKEH_URL=https://srrweb.cc.lehigh.edu/chosen-prefix/main
```

Restart the Flask service after changing `SINDY_BOKEH_URL`.

## 4. Verify before merging or releasing

```bash
curl --fail --silent --output /dev/null http://127.0.0.1:5006/sindy-bokeh/main
curl --fail --silent --output /dev/null https://srrweb.cc.lehigh.edu/SINDy
curl --fail --silent --output /dev/null https://srrweb.cc.lehigh.edu/sindy-bokeh/main
```

Finally, open `https://srrweb.cc.lehigh.edu/SINDy` and confirm that the Train,
Test, Predict, and Ensemble tabs appear and remain interactive. A rendered
page without working controls usually means the WebSocket proxy or allowed
origin is misconfigured.

## Rollback

Stop the SINDy Bokeh process, remove only the `/sindy-bokeh/` proxy rule introduced
for this module, and roll the parent Flask deployment back to its previous
commit. Other modules do not need to be changed.
