# SINDy.py

from tabs.ensemble_tab import ensemble_tab_layout
from tabs.test_tab import test_tab_layout
from tabs.predict_tab import predict_tab_layout
from tabs.train_tab import train_tab_layout
from engine.sindy_model import SINDyEngine
from bokeh.models import Tabs, TabPanel
from bokeh.plotting import curdoc
import os
import sys

# Match the existing STEM modules: the public Bokeh route is derived from this
# filename, so ``bokeh serve SINDy.py`` exposes the app at /SINDy.
sys.path.append(os.path.dirname(__file__))


def build_app():
    # --- 1. Session Isolation ---
    engine = SINDyEngine()
    trained_model_storage = {}

    # --- 2. Instantiate Tabs ---
    layout_train = train_tab_layout(engine, trained_model_storage)
    layout_test,    update_test = test_tab_layout(
        engine, trained_model_storage)
    layout_predict, update_predict = predict_tab_layout(
        engine, trained_model_storage)
    layout_ensemble, update_ensemble = ensemble_tab_layout(
        engine, trained_model_storage)

    # --- 3. TabPanel ---
    tab1 = TabPanel(child=layout_train,   title="🏋 Train & Validate")
    tab2 = TabPanel(child=layout_test,    title="🧪 Test")
    tab3 = TabPanel(child=layout_predict, title="🔮 Predict")
    tab4 = TabPanel(child=layout_ensemble, title="🎲 Ensemble")

    project_tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])

    # --- 4. Change Tab Logic ---
    def on_tab_change(attr, old, new):
        if new == 1:
            update_test()
        elif new == 2:
            update_predict()
        elif new == 3:
            update_ensemble()

    project_tabs.on_change('active', on_tab_change)

    return project_tabs


# Create a separate document for every Bokeh session.
curdoc().add_root(build_app())
curdoc().title = "SINDy Module - Lab Dashboard"
