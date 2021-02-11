# import unittest
# from fastapi.testclient import TestClient

# from src.app import app

# client = TestClient(app)


# class PredictRouteTestCase(unittest.TestCase):
#     def test_read_main(self):
#         with open("tests/test_data/detector_data/images_test/EAF_ko_16.jpg", "rb") as f:
#             response = client.post(
#                 "/predict",
#                 data={'file': f},
#             )
#             self.assertEqual(response.status_code, 200)