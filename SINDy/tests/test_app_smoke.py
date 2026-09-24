import unittest

from bokeh.models import Plot, Tabs

from main import build_app


class AppSmokeTests(unittest.TestCase):
    def test_build_app_wires_all_four_tabs(self):
        app = build_app()
        self.assertIsInstance(app, Tabs)
        self.assertEqual(
            [panel.title for panel in app.tabs],
            ["🏋 Train & Validate", "🧪 Test", "🔮 Predict", "🎲 Ensemble"],
        )
        self.assertTrue(all(plot.renderers for plot in app.select(
            {"type": Plot})))


if __name__ == "__main__":
    unittest.main()
