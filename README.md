# Python based Interactive Scientific Visualization

Python Based Interactive Scientific Visualization: 

We describe various science and engineering problems via interactive web based visualizations. An interactive web based approach will increase understanding of science and engineering problems as users can interact with parameters in a live setting.

Bokeh is used as the base package for visualizations.

### Prerequisites

Need to install python 3.8x or higher and add it to path on your system. You can check it by opening cmd.exe or a terminal and typing:

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

If already running an interactive visualization default port will be unavailable. You can specify port usin --port

```
$bokeh serve --show <filename.py> --port 5010
```

The interactive webpage can be found here: 

* [Sequential_Reactions](http://srrweb.cc.lehigh.edu/sliders_reaction_kinetics) - Sequential reactions example

## Built With

* [Bokeh](https://docs.bokeh.org/en/latest/) - The visualization library used

## Authors

* **Raghuram Thiagarajan** - *Initial work* - [swamilikes2code](https://github.com/swamilikes2code)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License

## Acknowledgments

* Python Community
* PyData Ann Arbor Meetup
* Bokeh Visualization Library

