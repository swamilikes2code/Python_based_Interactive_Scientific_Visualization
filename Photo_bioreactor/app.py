from flask import Flask, render_template
from bokeh.embed import components
from bokeh.resources import CDN
import os

# Import the Bokeh application creation function from your modified BO_final.py
from BO_final import create_bokeh_layout_for_flask

# Initialize Flask app, specifying template and static folders
app = Flask(__name__, template_folder='templates', static_folder='static')

# Route for the main page that embeds the Bokeh app
@app.route('/')
@app.route('/Bayesian_Optimization') # Added this route for your dropdown menu link
def index():
    # Call the function from BO_final.py to get the Bokeh layout object
    bokeh_layout = create_bokeh_layout_for_flask()

    # Generate Bokeh script and div components
    script, div = components(bokeh_layout)

    # Render the HTML template, passing Bokeh components and CDN resources
    return render_template('BO_HTML.html', bokeh_script=script, bokeh_div=div, bokeh_cdn=CDN.render())

# Route for serving static files (Flask automatically handles 'static' folder)
# This explicit route is good practice for clarity but can often be omitted if
# all static files are in the 'static' folder and accessed via url_for('static', ...)
@app.route('/static/<path:filename>')
def static_files(filename):
    return app.send_static_file(filename)

# Placeholder routes for other modules linked in your HTML
# If these modules are complex Flask apps, they would need their own Python files and routes.
# If they are just static HTML pages, ensure they are in the 'static' folder and linked via url_for('static_files', ...)
@app.route('/biodegradability_classification')
def biodegradability_classification():
    return "<h1>Biodegradability Classification Module Coming Soon!</h1><p>This is a placeholder page.</p>"

@app.route('/catalysis_data_interactive_visualization')
def catalysis_data_interactive_visualization():
    return "<h1>Catalysis Data Science Module Coming Soon!</h1><p>This is a placeholder page.</p>"

@app.route('/DLT')
def dlt():
    return "<h1>Digital Lab Twin Module Coming Soon!</h1><p>This is a placeholder page.</p>"

@app.route('/DAC')
def dac():
    return "<h1>Direct Air Captures Module Coming Soon!</h1><p>This is a placeholder page.</p>"

@app.route('/sliders_reaction_kinetics')
def sliders_reaction_kinetics():
    return "<h1>Reaction Kinetics Module Coming Soon!</h1><p>This is a placeholder page.</p>"

@app.route('/SIR')
def sir():
    return "<h1>SEIR Infectious Diseases Module Coming Soon!</h1><p>This is a placeholder page.</p>"

@app.route('/ZECC')
def zecc():
    return "<h1>Zero Energy Cooling Chamber Module Coming Soon!</h1><p>This is a placeholder page.</p>"

# Route for your "about" page
# Assuming about.html is a simple static HTML file placed directly in the 'static' folder.
@app.route('/about')
def about():
    return app.send_static_file('about.html')


if __name__ == '__main__':
    app.run(debug=True)