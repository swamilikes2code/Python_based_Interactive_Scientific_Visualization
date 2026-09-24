import unittest

import numpy as np
import pandas as pd
from bokeh.models import Button, CheckboxButtonGroup, DataTable, Plot

from engine.sindy_model import SINDyEngine
from tabs.ensemble_tab import _ensemble_label
from tabs.predict_tab import _default_initial_condition
from tabs.test_tab import (test_tab_layout, _model_variable_count,
                           _validate_test_columns)
from tabs.train_tab import train_tab_layout, _compact_trajectory_label


class UIContractTests(unittest.TestCase):
    def test_predict_uses_primary_ic_from_multi_trajectory_run(self):
        saved = {"initial_conditions": [[1.0, 2.0], [3.0, 4.0]]}
        self.assertEqual(_default_initial_condition(saved, 2), [1.0, 2.0])

    def test_predict_accepts_legacy_flat_ic(self):
        saved = {"initial_conditions": [1.0, 2.0]}
        self.assertEqual(_default_initial_condition(saved, 2), [1.0, 2.0])

    def test_ensemble_label_has_one_canonical_format(self):
        self.assertEqual(
            _ensemble_label(1, 2, 100),
            "Run #1 - Ensemble #2 - n = 100",
        )

    def test_trajectory_button_label_keeps_ic_and_distinctive_suffix(self):
        self.assertEqual(
            _compact_trajectory_label("duffing_train_1.csv", 0),
            "IC 1 · duffing_train_1",
        )
        compact = _compact_trajectory_label(
            "very_long_experiment_initial_condition_05.csv", 4)
        self.assertTrue(compact.startswith("IC 5 · very_long_"))
        self.assertTrue(compact.endswith("dition_05"))
        self.assertIn("…", compact)

    def test_test_columns_require_training_order(self):
        correct = pd.DataFrame({"t": [0, 1], "x": [1, 2], "v": [0, 1]})
        reversed_columns = pd.DataFrame(
            {"t": [0, 1], "v": [0, 1], "x": [1, 2]})

        self.assertIsNone(_validate_test_columns(correct, ["x", "v"], 2))
        self.assertIn(
            "order mismatch",
            _validate_test_columns(reversed_columns, ["x", "v"], 2),
        )

    def test_test_columns_reject_wrong_names(self):
        renamed = pd.DataFrame(
            {"t": [0, 1], "position": [1, 2], "velocity": [0, 1]})
        self.assertIn(
            "names mismatch",
            _validate_test_columns(renamed, ["x", "v"], 2),
        )

    def test_legacy_model_dimension_falls_back_to_dataframe_width(self):
        class LegacyModel:
            pass

        data = pd.DataFrame({"t": [0, 1], "x": [1, 2], "v": [0, 1]})
        self.assertEqual(_model_variable_count(LegacyModel(), data), 2)

    def test_test_button_callback_executes_with_a_selected_model(self):
        class StationaryModel:
            n_features_in_ = 4

            def predict(self, X):
                return np.zeros_like(X)

        storage = {
            1: {
                "model_instance": StationaryModel(),
                "system_name": "cs_train_data.csv",
                "feature_names": ["x1", "v1", "x2", "v2"],
            }
        }
        layout, update_model_list = test_tab_layout(SINDyEngine(), storage)
        update_model_list()
        test_button = next(
            button for button in layout.select({"type": Button})
            if button.label == "TEST"
        )

        # Invoke the registered handler to exercise the complete nested
        # callback: CSV validation, dimension lookup, solve_ivp, plotting,
        # and metrics assembly.
        test_button._event_callbacks["button_click"][0]()

    def test_multi_trajectory_buttons_filter_the_matching_fit_line(self):
        t1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.0, 1.0, 2.0])
        X1 = np.array([[1.0], [0.5], [0.0]])
        X2 = np.array([[2.0], [1.0], [0.0]])
        storage = {
            1: {
                "feature_names": ["x"],
                "warning": None,
                "plot_data": {
                    "t": np.concatenate([t1, t2]),
                    "X": np.vstack([X1, X2]),
                    "trajectories": [(X1, t1), (X2, t2)],
                    "trajectory_labels": ["duffing_train_1.csv",
                                          "duffing_train_2.csv"],
                    "train_idx": np.array([0, 1, 3, 4]),
                    "val_idx": np.array([2, 5]),
                    "ic_sims": [
                        {"t": t1, "x_sim": X1,
                         "label": "duffing_train_1.csv"},
                        {"t": t2, "x_sim": X2,
                         "label": "duffing_train_2.csv"},
                    ],
                },
            }
        }
        layout = train_tab_layout(SINDyEngine(), storage)
        history = next(iter(layout.select({"type": DataTable})))
        history.source.data = {
            "run": [1], "system": ["Duffing"], "split": ["Random Block"],
            "lib": ["Polynomial"], "poly": [3], "thr": [0.1],
            "train_metrics": [""], "val_metrics": [""],
            "rmse_diff": [""], "equations": [""],
        }
        history.source.selected.indices = [0]

        trajectory_toggle = next(
            toggle for toggle in layout.select({"type": CheckboxButtonGroup})
            if toggle.labels and toggle.labels[0].startswith("IC 1 ·")
        )
        self.assertEqual(len(trajectory_toggle.labels), 2)
        self.assertEqual(trajectory_toggle.active, [0, 1])

        model_plot = next(
            plot for plot in layout.select({"type": Plot})
            if plot.title.text.startswith("Model Result — Run #1")
        )
        fit_lines = [
            renderer for renderer in model_plot.renderers
            if "trajectory" in renderer.data_source.data
        ]
        self.assertEqual(len(fit_lines), 2)
        fit_by_trajectory = {
            renderer.data_source.data["trajectory"][0]: renderer
            for renderer in fit_lines
        }
        self.assertEqual(
            fit_by_trajectory["duffing_train_1.csv"].glyph.line_dash, [])
        self.assertEqual(
            fit_by_trajectory["duffing_train_2.csv"].glyph.line_dash, [6])

        trajectory_toggle.active = [0]
        visibility = {
            renderer.data_source.data["trajectory"][0]: renderer.visible
            for renderer in fit_lines
        }
        self.assertTrue(visibility["duffing_train_1.csv"])
        self.assertFalse(visibility["duffing_train_2.csv"])

        # Empty selection is a supported state: every trajectory layer must
        # disappear, leaving an intentionally blank model plot.
        trajectory_toggle.active = []
        self.assertTrue(all(not renderer.visible for renderer in fit_lines))
        data_renderers = [
            renderer for renderer in model_plot.renderers
            if "trajectory" not in renderer.data_source.data
        ]
        self.assertTrue(all(
            renderer.glyph.fill_alpha == 0 for renderer in data_renderers))

        # Selection order defines the focused solid line. IC2 stays focused
        # when IC1 is added later, then IC1 is promoted when IC2 is hidden.
        trajectory_toggle.active = [1]
        self.assertEqual(
            fit_by_trajectory["duffing_train_2.csv"].glyph.line_dash, [])
        trajectory_toggle.active = [0, 1]
        self.assertEqual(
            fit_by_trajectory["duffing_train_2.csv"].glyph.line_dash, [])
        self.assertEqual(
            fit_by_trajectory["duffing_train_1.csv"].glyph.line_dash, [6])
        trajectory_toggle.active = [0]
        self.assertEqual(
            fit_by_trajectory["duffing_train_1.csv"].glyph.line_dash, [])

        show_all_button = next(
            button for button in layout.select({"type": Button})
            if button.label == "SHOW ALL"
        )
        show_all_button._event_callbacks["button_click"][0]()
        self.assertEqual(trajectory_toggle.active, [0, 1])

    def test_new_training_run_is_selected_and_delete_is_enabled(self):
        storage = {}
        layout = train_tab_layout(SINDyEngine(), storage)
        buttons = list(layout.select({"type": Button}))
        train_button = next(button for button in buttons
                            if button.label == "TRAIN")
        delete_button = next(button for button in buttons
                             if button.label == "DELETE")
        history = next(iter(layout.select({"type": DataTable})))

        self.assertTrue(delete_button.disabled)
        train_button._event_callbacks["button_click"][0]()

        self.assertEqual(history.source.selected.indices, [0])
        self.assertFalse(delete_button.disabled)
        self.assertIn(1, storage)


if __name__ == "__main__":
    unittest.main()
