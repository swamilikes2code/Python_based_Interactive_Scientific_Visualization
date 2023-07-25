
from bokeh.models import Select, Div, CustomJS
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.resources import Resources



font_options = ["Arial", "San Serif", "Times New Roman", "OpenDyslexic"]
fontAccess = Select(title="Font Options:", value="Arial", options= font_options, height = 60, width = 300)# Student chooses the loss function
def font_Callback(attr, old, new):
    if(new == "Arial"):
        intro.styles = {"font-family": "Arial"}
    elif(new == "San Serif"):
        intro.styles = {"font-family": "San Serif"}
    elif(new == "OpenDyslexic"):
        intro.styles = {"font-family": "opendyslexic"}
    else:
        intro.styles = {"font-family": "Times New Roman"}
    
# fontAccess.on_change('value', font_Callback)


font_size_options = ["100%", "110%", "120%", "130%", "140%", "150%", "160%", "170%", "180%", "190%", "200%"]
sizeAccess = Select(title="Text Size Options:", value="100", options= font_size_options, height = 60, width = 300)# Student chooses the loss function

def size_Callback(attr, old, new, font):
    intro.styles = {'font-size': new, "font-family": font.value}

# sizeAccess.on_change('value', size_Callback, fontAccess)

# Define the callback function for font and size changes
def font_size_callback(attr, old, new):
    selected_font = fontAccess.value
    selected_size = sizeAccess.value
    
    intro.styles = {'font-size': selected_size, "font-family": selected_font}

# Attach the callback functions to the value change events of the font and size widgets
fontAccess.on_change('value', font_size_callback)
sizeAccess.on_change('value', font_size_callback)



css_rule = """
    @font-face {
    font-family: 'opendyslexic';
        src: url('fonts/OpenDyslexic-Regular.otf');
        font-style: normal;
        font-weight: normal;
    } 

    @font-face {
        font-family: 'opendyslexic';
        src: url('fonts/OpenDyslexic-Italic.otf');
        font-style: italic;
        font-weight: normal;
    }

    @font-face {
        font-family: 'opendyslexic';
        src: url('fonts/OpenDyslexic-Bold.ttf');
        font-weight: bold;
        font-style: normal;
    } 

    @font-face {
        font-family: 'opendyslexic';
        src: url('fonts/OpenDyslexic-BoldItalic.ttf');
        font-weight: bold;
        font-style: italic;
    } 


    * {
        font-size:20px;
        line-height:200%;
        color: #191A42;
        font-family:opendyslexic !important;
    }
"""





# Add the CSS rule to the document head
# curdoc().add_root(CustomJS(code=css_rule))


intro = Div(text="""
       
        <h3>Simple Photobioreactor Summary</h3>
        <p>A photobioreactor is a container, like a fish tank, filled with water and special microscopic plants called algae. 
        It provides the algae with light, nutrients, and carbon dioxide to help them grow. 
        The algae use sunlight to make their own food through a process called photosynthesis. 
        The photobioreactor allows us to grow algae and use their biomass to produce clean and renewable energy,
        like biofuels. It also helps clean up the environment by absorbing harmful pollutants, such as carbon dioxide.
        In simple terms, a photobioreactor is a special container that helps tiny plants called algae grow using
        light, nutrients, and carbon dioxide to make clean energy and help the planet.(section for the paragraph).</p>
        
        <h3>Sliders</h3>
        <p> <b>To generate data, you can change the following values with the coresponding slider</b>:<br>
            Initial Nitrate Concentration: Set the initial nitrate concentration in g/L (0.2 - 2)<br>
            Initial Biomass Concentration: Set the initial biomass concentration in g/L (0.2 - 2)<br>
            Inlet Flow: Control the inlet flow of nitrate into the reactor (0.001 - 0.015 L/h)<br>
            Inlet Concentration: Inlet concentration of nitrate feed to the reactor (5 - 15 g/L)<br>
            Light Intensity: Control the light intensity in the range of 100 - 200 umol/m2-s </p>

         <h3>Hover Tool</h3>
        <p> Hover the cursor over the line and you will be able to the element name, time, and  element concentration <br>
            <b><u>Note</u></b>: the Lutien concentration is 1000 times greater than what the actual concentration is, so that you are able to see the Lutine curve</p>
        
         <h3>Reset Button</h3>
        <p> This Button will reset the graph to the original position based on the initial conditions before the sliders were changed</p>
        
         <h3>Run Button</h3>
        <p> This Button will take the slider conditions that you have and will create a new plot based on those new conditions<br>
        <b><u>Note</u></b>: There are two run buttons the run button on the Train Tab changes the graph in the Train tab and the Optimize tab, <br>
        the run button on the Evaluate Tab on will only change the graph of the evaluate tab</p>
        
         <h3>Export Button</h3>
        <p> This Button will take the data points of the Time, Nitrate Concentration, Biomass concentration, and Lutine concentration<br>
        and put them in a csv file and this csv file will be located in your downloads folder the file will be named "exported_data_{timestamp}.csv"<br>
        the timestamp is the current time and will be formated as year-month-day-hour-minuete-second</p>
        
         <h3>Help Button</h3>
        <p> In the Train tab you can see little question mark buttons next to the interactive elements, these buttons will give you <br>
        information on what this tool is used for and how it will can change your graph</p>
        
        
        
    """)


choose = row (fontAccess, sizeAccess)
test = column(choose, intro)
curdoc().add_root(test, )

