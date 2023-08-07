# flask app

from flask import Flask, render_template
from bokeh.client import pull_session
from bokeh.embed import server_document
from flask import send_from_directory

#NOTE: this version of the flask_app is simply to guide a new user through the process of setting up a flask app.
#It is not meant to be used as a final product.  For a more complete version of the flask_app, see the flask_app.py in the flask_server_setup folder.
#To run this example with the example Bokeh, create two separate terminals. In the first, navigate to the python folder.
#In the second, navigate to the app folder.  In the first terminal, run bokeh serve --allow-websocket-origin=127.0.0.1:8080 bokehExample.py.  
#In the second, run python flask_app.py.

# instantiate the flask app
app = Flask(__name__, static_url_path="/app/static")


# create index page function
@app.route("/app", methods=["GET"])
def index():
    return render_template("index.html")
#also route root to index page
@app.route("/", methods=["GET"])
def index2():
    return render_template("index.html")

#NOTE: most pages have been removed for simplicity.  See the flask_app.py in the flask_server_setup folder for a more complete version.

#create a new route for our example HTML page
@app.route("/example", methods=["GET"])
def example():
    #tell the server where to expect the bokehModule to be running. In this case, it is running on the same machine as the flask app, so we use localhost.
    bokeh_script_exampleModel = server_document(url="http://localhost:5006/bokehExample")
    #tell flask which template to render, and point it to the bokeh_script_exampleModel variable we just created
    return render_template("example.html", bokeh_script_exampleModel=bokeh_script_exampleModel)

# run the app
if __name__ == "__main__":
    app.run(port=8080)
