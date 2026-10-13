# Flask Server Setup

A quick example showing set up of Flask server consisting of ZECC and reaction_kinetics examples.


### Prerequisites

We are using [Flask](https://pypi.org/project/Flask/) to embed multiple interactive visualizations in a HTML webpage. You can install Flask using:

```
$pip install flask
```

### Setup

Step 1: Open a command line. Navigate to the repository folder reaction_kinetics in your local machine. Run the reaction kinetics example using:

```
bokeh serve --allow-websocket-origin=localhost:8080 sliders_reaction_kinetics.py 
```

Step 2: Open a command line. Navigate to the repository folder ZECC in your local machine. Run the ZECC example using:

```
bokeh serve --allow-websocket-origin=localhost:8080 --port=5007 ZECC.py 
```

Step 3: Run the Flask server using:
```
python flask_app.py 
```

Step 4: Navigate to localhost:8080 to see the HTML setup.

Below is a link to the screenshot of command line commands in the repository.

![Screesnshot of CMD](https://github.com/swamilikes2code/Python_based_Interactive_Scientific_Visualization/blob/master/flask_server_setup/images/flask_server_setup_command_line.PNG)

The final output on your local machine should appear as:

![Screesnshot of Final Output](https://github.com/swamilikes2code/Python_based_Interactive_Scientific_Visualization/blob/master/flask_server_setup/images/final_server_output.PNG)

## SINDy module

The SINDy expert system is mounted at `/SINDy` in the parent Flask site. Run
its Bokeh backend from the module directory so its built-in `data/` paths
resolve correctly:

```bash
cd SINDy
pip install -r requirements.txt
bokeh serve main.py --port 5006 \
  --prefix /sindy-bokeh \
  --allow-websocket-origin=localhost:8080
```

In a second terminal, start the parent Flask application:

```bash
export SINDY_BOKEH_URL=http://localhost:5006/sindy-bokeh/main
python flask_server_setup/app/flask_app.py
```

Then open `http://localhost:8080/SINDy`. Flask owns the public page at
`/SINDy`; Bokeh is deliberately mounted at `/sindy-bokeh/main` so the two services
never compete for the same URL.

In production, the Flask page is `https://srrweb.cc.lehigh.edu/SINDy` and its
embedded backend defaults to `https://srrweb.cc.lehigh.edu/sindy-bokeh/main`. The
included launcher uses the public host, `/sindy-bokeh` prefix, and port `5006` by
default:

```bash
./SINDy/serve_parent_site.sh
```

Override `SINDY_PORT`, `SINDY_ADDRESS`, `SINDY_URL_PREFIX`, or
`SINDY_WEBSOCKET_ORIGIN` when the server uses different values.

The web server must proxy the complete `/sindy-bokeh/` prefix, including WebSocket
upgrades, to the Bokeh process. Do not proxy `/SINDy` to Bokeh; that route must
continue to reach Flask. Set `SINDY_BOKEH_URL` only when the public endpoint
differs from the default. Full deployment and verification steps are in
[`../SINDy/DEPLOYMENT.md`](../SINDy/DEPLOYMENT.md).
