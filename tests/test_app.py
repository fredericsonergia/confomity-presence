import unittest
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app.app)


class PresenceRouteTestCase(unittest.TestCase):
    def test_presence_main(self):
        with open("tests/test_images/EAF_ko_16.jpg", "rb") as f:
            response = client.post(
                "/presence",
                files={"file": ("EAF_ko_16.jpg", f, "multipart/form-data")},
            )
            self.assertEqual(response.status_code, 200)


class ConformityRouteTestCase(unittest.TestCase):
    def test_conformity_main(self):
        with open("tests/test_images/EAF_ko_16.jpg", "rb") as f:
            response = client.post(
                "/conformity",
                files={"file": ("EAF_ko_16.jpg", f, "multipart/form-data")},
            )
            self.assertEqual(response.status_code, 200)
