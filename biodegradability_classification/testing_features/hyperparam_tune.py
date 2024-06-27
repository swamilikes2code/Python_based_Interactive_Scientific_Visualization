from bokeh.models import Slider, Select, Checkbox
from bokeh.layouts import row, column
from bokeh.io import curdoc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# global my_alg
my_alg = 'Support Vector Classification'
global model
# model = "test"

# hyperparameter tuning widgets
hp_slider = Slider()
hp_select = Select()
hp_toggle = Checkbox()
hp_toggle.margin = (24, 10, 24, 10)


if my_alg == 'Decision Tree':
    #hyperparameters are 
    # splitter strategy (splitter, best vs. random, select)
    # max_depth of tree (max_depth, int slider)
    model = DecisionTreeClassifier()

    # testing purposes
    def print_vals():
        print("slider", hp_slider.value)
        print("switch", hp_toggle.active)
        print("model", model.max_depth)
        print("model splitter", model.splitter)

    # if switch on, max_depth = None
    def hp_toggle_callback(attr, old, new):
        if new == True:
            hp_slider.update(disabled = True, bar_color = 'black', show_value = False)
            # hp_slider.disabled = True
            # hp_slider.bar_color = 'gray'
            # hp_slider.show_value = False
            model.max_depth = None
        elif new == False:
            hp_slider.update(disabled = False, bar_color = '#e6e6e6', show_value = True)
            model.max_depth = hp_slider.value
        print_vals()

    def hp_slider_callback(attr, old, new):
        if hp_slider.disabled == True:
            return
        model.max_depth = new
        print_vals()

    def hp_select_callback(attr, old, new):
        model.splitter = new
        print_vals()

    hp_slider.update(
        title = "Max Depth of Tree",
        start= 0,
        end = 15,
        value = 2,
        step = 1
    )
    hp_slider.on_change('value', hp_slider_callback)

    hp_toggle.update(
        label = "None",
        visible = True,
        active = False
    )
    hp_toggle.on_change('active', hp_toggle_callback)

    hp_select.update(
        title = "Splitter strategy",
        value = "best",
        options = ["best", "random"]
    )
    hp_select.on_change('value', hp_select_callback)

elif my_alg == 'K-Nearest Neighbors':
    #hyperparameters are 
    # K (n_neighbors, int slider)
    # weights (weights, uniform vs. distance, select)
    model = KNeighborsClassifier()

    # testing
    def print_vals():
        print("slider", hp_slider.value)
        print("n_neighbors", model.n_neighbors)
        print("weights", model.weights)

    def hp_slider_callback(attr, old, new):
        model.n_neighbors = new
        print_vals()
    
    def hp_select_callback(attr, old, new):
        model.weights = new
        print_vals()

    hp_slider.update(
        title = "Number of neighbors",
        start = 0,
        end = 30,
        value = 5,
        step = 5
    )
    hp_slider.on_change('value', hp_slider_callback)

    hp_toggle.visible = False

    hp_select.update(
        title = "Weights",
        value = "uniform",
        options = ["uniform", "distance"]
    )
    hp_select.on_change('value', hp_select_callback)

elif my_alg == 'Support Vector Classification':
    #hyperparameters are 
    # loss (loss, hinge vs. squared_hinge, select) 
    # the max iterations to be run (max_iter, int slider)
    model = LinearSVC()

    def print_vals():
        print("slider", hp_slider.value)
        print("max iter", model.max_iter)
        print("loss func", model.loss)

    def hp_slider_callback(attr, old, new):
        model.max_iter = new
        print_vals()
    
    def hp_select_callback(attr, old, new):
        model.loss = new
        print_vals()

    hp_slider.update(
        title = "Maximum iterations", #default is 1000
        start = 500,
        end = 1500,
        value = 1000,
        step = 100
    )
    hp_slider.on_change('value', hp_slider_callback)

    hp_toggle.visible = False

    hp_select.update(
        title = "Loss function",
        value = "squared_hinge",
        options = ["squared_hinge", "hinge"]
    )
    hp_select.on_change('value', hp_select_callback)

l = column(row(hp_slider, hp_toggle), hp_select)
curdoc().add_root(l)