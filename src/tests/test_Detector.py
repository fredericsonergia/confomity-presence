import unittest
import sys 

from Detector.Detector import ModelBasedDetector

class ModelBasedDetectorTestCase(unittest.TestCase):
    def setUp(self):
        detector1 = ModelBasedDetector.from_pretrained(data_path='../Data/EAF_real', batch_size=10, base_model='ssd_512_mobilenet1.0_custom', save_prefix='ssd_512')
        detector2 = ModelBasedDetector.from_finetuned('../models/ssd_512_best.params',data_path='../Data/EAF_real', batch_size=batch_size, base_model='ssd_512_mobilenet1.0_custom', save_prefix='ssd_512')

    def test_from_pretrained_att(self):
        self.assertEqual(self.detector1.net, "<class 'gluoncv.model_zoo.ssd.ssd.SSD'>")

if __name__ == '__main__':
    unittest.main()