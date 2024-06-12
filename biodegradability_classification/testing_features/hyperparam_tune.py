from bokeh.models import Slider, Select
from bokeh.layouts import row, column
from bokeh.io import curdoc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# global my_alg
my_alg = 'K-Nearest Neighbors'
global model
# model = "test"

slider = Slider()
select = Select()


if my_alg == 'Decision Tree':
    #hyperparameters are 
    # splitter strategy (splitter, best vs. random, select)
    # max_depth of tree (max_depth, int slider)
    model = DecisionTreeClassifier()

    slider.title = "Max Depth of Tree" #can link to the model's hyperparam
    slider.start= 0
    slider.end = 15
    slider.value = 2
    slider.step = 1

    select.title = "Splitter strategy"
    select.value = "best"
    select.options = ["best", "random"]

elif my_alg == 'K-Nearest Neighbors':
    #hyperparameters are 
    # K (n_neighbors, int slider)
    # weights (weights, uniform vs. distance, select)
    model = KNeighborsClassifier()

    slider.title = "Number of neighbors"
    slider.start = 0
    slider.end = 30
    slider.value = 5
    slider.step = 5

    select.title = "Weights"
    select.value = "uniform"
    select.options = ["uniform", "distance"]

elif my_alg == 'Support Vector Machine':
    #hyperparameters are 
    # loss (loss, hinge vs. squared_hinge, select) 
    # the max iterations to be run (max_iter, int slider)
    model = LinearSVC()

    slider.title = "Maximum iterations"
    slider.start = 500
    slider.end = 1500
    slider.value = 1000
    slider.step = 100

    select.title = "Loss function"
    select.value = "squared_hinge"
    select.options = ["squared_hinge", "hinge"]


# for checking that assignment worked

# def cb(attr, old, new):
#     print(new)
#     print(model)

# slider.on_change('value', cb)

l = column(slider, select)
curdoc().add_root(l)