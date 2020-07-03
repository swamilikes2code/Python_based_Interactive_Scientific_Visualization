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
