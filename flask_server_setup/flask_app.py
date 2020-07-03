#flask app

from flask import Flask, render_template
from bokeh.client import pull_session
from bokeh.embed import server_document

#instantiate the flask app
app = Flask(__name__)

#create index page function
@app.route("/", methods=['GET'])
def index():
    bokeh_script_reaction_kinetics = server_document(url="http://localhost:5006/sliders_reaction_kinetics")
    bokeh_script_ZECC = server_document(url="http://localhost:5007/ZECC")
    return render_template("index.html", bokeh_script_reaction_kinetics=bokeh_script_reaction_kinetics, bokeh_script_ZECC=bokeh_script_ZECC)

#run the app
if __name__ == "__main__":
    app.run(port=8080)
