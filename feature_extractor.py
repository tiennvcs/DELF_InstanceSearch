import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf.protos import delf_config_pb2
from delf import feature_io
from delf import utils
from delf.examples import extractor


class FeatureExtractor():
    def __init__(self, config_path):
        
        config = delf_config_pb2.DelfConfig()
        with tf.io.gfile.GFile(config_path, 'r') as f:
            text_format.Merge(f.read(), config)

        self.extractor_fn = extractor.MakeExtractor(config)
        

    def extract(self, img):
        
        extracted_features = self.extractor_fn(img)
        # locations_out = extracted_features['local_features']['locations']
        # descriptors_out = extracted_features['local_features']['descriptors'] ## --> [1000, 40]
        # feature_scales_out = extracted_features['local_features']['scales']
        # attention_out = extracted_features['local_features']['attention']
        return extracted_features['local_features']


if __name__ == '__main__':
    
    img_path = './static/database/images/nhatho1.jpg'
    im = np.array(utils.RgbLoader(img_path))

    config_path = 'config/delf_config.pbtxt'
    model = FeatureExtractor(config_path)
    model.extract(im)
