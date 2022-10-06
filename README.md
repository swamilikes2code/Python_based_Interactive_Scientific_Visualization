# Python based Interactive Scientific Visualization

Python Based Interactive Scientific Visualization: 

We describe various science and engineering problems via interactive web based visualizations. An interactive web based approach will increase understanding of science and engineering problems as users can interact with parameters in a live setting.

Bokeh is used as the base package for visualizations.

### Prerequisites

Need to install [python 3.8](https://www.python.org/downloads/) and add it to path on your system before exiting the installation process. You can check it by opening cmd.exe or a terminal and typing:

```
$python --version
```

numpy, scipy and bokeh are required libraries. To install these use the command:

```
$pip install numpy
$pip install scipy
$pip install bokeh
```

### Running interactive HTML

You can create an interactive webpage from any example using:

```
$bokeh serve --show <filename.py>
```

If already running an interactive visualization default port will be unavailable. You can specify port using --port

```
$bokeh serve --show <filename.py> --port 5010
```

The interactive webpage can be found here: 

* [Zero Energy Cooling Chamber](https://srrweb.cc.lehigh.edu/app/ZECC) - Zero Energy Cooling Chamber example

* [Susceptible Exposed Infected Recovered Model](https://srrweb.cc.lehigh.edu/app/SIR) - Susceptible Exposed Infected Recovered Model example

* [Sequential_Reactions](https://srrweb.cc.lehigh.edu/app/sliders_reaction_kinetics) - Sequential Reactions example

* [Catalysis Data Science Model](https://srrweb.cc.lehigh.edu/app/catalysis_data_interactive_visualization) - Catalysis Data Science Model example

* [Direct Air Capture](https://srrweb.cc.lehigh.edu/app/DAC) - Direct Air Capture example

## Built With

* [Bokeh](https://docs.bokeh.org/en/latest/) - The visualization library used

* [Flask](https://pypi.org/project/Flask/) - Used to embed interactive Bokeh applications in a HTML

* [Scikit-learn](https://scikit-learn.org/stable/index.html) - The regression library used

* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - Used to generate correlation between attributes

## Authors

* Class of 2021

    - **Timothy Odom** - *Core Developer* - [too223](https://github.com/too223)
    
    - **Xu Chen** - *Core Developer* - [xuc323](https://github.com/xuc323)

    - **Neel Surya** - *Design, Implementation and UI/UX Developer* - [Neel-Surya](https://github.com/Neel-Surya)

    - **Alex Outkou** - *UI/UX and HTML Developer* - [ado323](https://github.com/ado323)
    
* Class of 2020

    - **Anna Moragne** - *Core Developer* - [annamoragne](https://github.com/annamoragne)

    - **Brian Lucas** - *Design, Implementation and UI/UX Developer* - [cariocabrian](https://github.com/cariocabrian)

* **Raghuram Thiagarajan** - *Project Lead* - [swamilikes2code](https://github.com/swamilikes2code)

* **Srinivas Rangarajan** - *Principal Investigator* - [Faculty Website at Lehigh](https://engineering.lehigh.edu/faculty/srinivas-rangarajan)

See also the list of [contributors](https://github.com/swamilikes2code/Python_based_Interactive_Scientific_Visualization/graphs/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License

## Acknowledgments

* [Bokeh_CorrelationMatrix](https://github.com/raghavsikaria/Bokeh_CorrelationMatrix)
* Python Community
* PyData Ann Arbor Meetup
* Bokeh Visualization Library
