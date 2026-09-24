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
  --allow-websocket-origin=localhost:8080
```

In a second terminal, start the parent Flask application:

```bash
export SINDY_BOKEH_URL=http://localhost:5006/main
python flask_server_setup/app/flask_app.py
```

Then open `http://localhost:8080/SINDy`. In production, set
`SINDY_BOKEH_URL` to the public Bokeh service URL and allow the parent
website's hostname in the Bokeh WebSocket origin list. The Bokeh service and
Flask service must be deployed separately because the interactive callbacks
run on the Bokeh server.
