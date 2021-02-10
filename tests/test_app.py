import unittest
from fastapi.testclient import TestClient

class ModelBasedDetectorTestCase(unittest.TestCase):
    
    def setup(self):
        self.f = open("./test_images", "r")

    def test_read_main(self):
        response = 200
        self.assertEqual(response.status_code,200)

    def tearDown(self):
        self.f.close()