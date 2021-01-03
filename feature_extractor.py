# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model
# import numpy as np

# See https://keras.io/api/applications/ for details

# class FeatureExtractor:
#     def __init__(self):
#         base_model = VGG16(weights='imagenet')
#         self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

#     def extract(self, img):
#         """
#         Extract a deep feature from an input image
#         Args:
#             img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)
#         Returns:
#             feature (np.ndarray): deep feature with the shape=(4096, )
#         """
#         img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
#         img = img.convert('RGB')  # Make sure img is color
#         x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
#         x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
#         x = preprocess_input(x)  # Subtracting avg values for each pixel
#         feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
#         return feature / np.linalg.norm(feature)  # Normalize

import numpy as np
from six.moves import range
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
        locations_out = extracted_features['local_features']['locations']
        descriptors_out = extracted_features['local_features']['descriptors'] ## --> [1000, 40]
        feature_scales_out = extracted_features['local_features']['scales']
        attention_out = extracted_features['local_features']['attention']
    
        return descriptors_out


if __name__ == '__main__':
    
    img_path = './static/database/images/nhatho1.jpg'
    im = np.array(utils.RgbLoader(img_path))

    config_path = 'config/delf_config.pbtxt'
    model = FeatureExtractor(config_path)
    model.extract(im)