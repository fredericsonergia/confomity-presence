from src.detector_utils.VOC_form import rename_img
import os
import unittest


class CliDataTestCase(unittest.TestCase):
    def setUp(self):
        try:
            os.rename("EAF_OK1.jpg", "a.jpg")
            os.rename("EAF_OK2.jpg", "a.jpg")

    def test_rename(self):
        name = ["EAF_OK1.jpg", "EAF_OK2.jpg"]
        rename_img("tests/test_data/rename_data/")
        rename = os.listdir("tests/test_data/rename_data/")
        self.assertIn(rename[0], name)
        self.assertIn(rename[1], name)


if __name__ == "__main__":
    unittest.main()
