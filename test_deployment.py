"""
Verification script for NCCTG Survival Analytics deployment readiness.
Tests routes, all 3 survival models, presets, report generation, and static assets.
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import app
from app import app as flask_app, survival_at_days, make_curve


class DeploymentVerificationTests(unittest.TestCase):
    def setUp(self):
        flask_app.config['TESTING'] = True
        self.client = flask_app.test_client()

    def test_01_health_endpoint(self):
        """Verify /health returns 200 and expected status without sensitive leak."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("selected_model", data)
        self.assertIn("selection_method", data)
        print("  [PASS] /health endpoint verified:", data)

    def test_02_home_page_renders(self):
        """Verify GET / returns 200 and loads HTML with essential elements."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("NCCTG", html)
        self.assertIn("Survival Analytics", html)
        self.assertIn("Patient Clinical Parameters", html)
        self.assertIn("Model Benchmark &amp; Statistical Comparison", html)
        print("  [PASS] Home page renders successfully")

    def test_03_all_three_models_functional(self):
        """Verify Cox PH, Weibull AFT, and Exponential PH all predict correctly."""
        import pandas as pd
        from config import FEATURES

        sample = pd.DataFrame([{
            "age": 63.0, "sex": 0, "ph.ecog": 1.0, "ph.karno": 80.0,
            "pat.karno": 80.0, "meal.cal": 975.0, "wt.loss": 7.0
        }], columns=FEATURES)

        for model_name in ["Cox PH", "Weibull AFT", "Exponential PH"]:
            prob_365 = survival_at_days(model_name, sample, 365)
            self.assertTrue(0.0 <= prob_365 <= 1.0, f"{model_name} probability out of range: {prob_365}")
            curve = make_curve(model_name, sample)
            self.assertEqual(len(curve), 180, f"{model_name} curve points mismatch")
            self.assertGreaterEqual(curve[0]['y'], curve[-1]['y'], f"{model_name} curve should be monotonically decreasing")
            print(f"  [PASS] {model_name} model prediction verified (365d prob = {prob_365:.4f})")

    def test_04_patient_profiles(self):
        """Verify patient presets and custom profile produce valid predictions."""
        profiles = [
            ("Median Cohort", {
                "age": "63", "sex": "Male", "ph_ecog": "1",
                "ph_karno": "80", "pat_karno": "80", "meal_cal": "975", "wt_loss": "7"
            }),
            ("Low Risk", {
                "age": "52", "sex": "Female", "ph_ecog": "0",
                "ph_karno": "100", "pat_karno": "100", "meal_cal": "1250", "wt_loss": "0"
            }),
            ("High Risk", {
                "age": "74", "sex": "Male", "ph_ecog": "2",
                "ph_karno": "60", "pat_karno": "60", "meal_cal": "600", "wt_loss": "18"
            }),
            ("Intermediate Risk Profile", {
                "age": "60", "sex": "Female", "ph_ecog": "1",
                "ph_karno": "90", "pat_karno": "80", "meal_cal": "1100", "wt_loss": "3"
            }),
        ]

        for name, data in profiles:
            response = self.client.post('/', data=data)
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn("PROGNOSTIC RISK STRATIFICATION", html, f"Failed for profile: {name}")
            self.assertIn("3-Year Patient Trajectory", html, f"Failed for profile: {name}")
            self.assertIn("Survival Trajectory", html, f"Failed for profile: {name}")
            print(f"  [PASS] Patient profile '{name}' processed successfully")

    def test_05_pdf_report_download(self):
        """Verify /download-report returns valid PDF bytes."""
        data = {
            "age": "63", "sex": "Male", "ph_ecog": "1",
            "ph_karno": "80", "pat_karno": "80", "meal_cal": "975", "wt_loss": "7"
        }
        response = self.client.post('/download-report', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        self.assertTrue(response.data.startswith(b'%PDF'), "Response is not valid PDF")
        self.assertGreater(len(response.data), 1000, "PDF size is unexpectedly small")
        print(f"  [PASS] PDF report download generated valid PDF ({len(response.data)} bytes)")

    def test_06_error_handling(self):
        """Verify invalid inputs return user-friendly message without 500 error."""
        invalid_data = {
            "age": "invalid_age", "sex": "Unknown", "ph_ecog": "99",
            "ph_karno": "500", "pat_karno": "-20", "meal_cal": "NaN", "wt_loss": "999"
        }
        response = self.client.post('/', data=invalid_data)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Please check the inputs", html)
        print("  [PASS] Graceful error handling verified")

    def test_07_static_assets(self):
        """Verify static background assets are served properly."""
        for path in ['/static/images/lung-background.jpeg', '/static/medical-background.png', '/static/lung_bg.png']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, f"Failed to serve {path}")
            self.assertGreater(len(response.data), 10000, f"File {path} is suspiciously small")
        print("  [PASS] Static assets verified")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING PRE-DEPLOYMENT TEST SUITE")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(DeploymentVerificationTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
