import os
import glob2
import numpy as np
from config.config import MAX_DESCRIPTOR
from delf import feature_io
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(path):
    """
    Return the list of images and list of relative features
    """
    types = ('*.png', '*.jpg') 
    
    features = []
    img_paths = []
    
    base_image_path = os.path.join(path, 'images/all_images')
    base_feature_path = os.path.join(path, 'features/40D/all_features')
    
    for type_img in types:
        img_paths.extend(sorted(glob2.glob(os.path.join(base_image_path, type_img))))
    for feature_path in sorted(glob2.glob(os.path.join(base_feature_path, '*.delf')))[:1000]:
        extracted_features = feature_io.ReadFromFile(feature_path)
        features.append(extracted_features)
    #assert len(img_paths) == len(features), "The number of features is not campatible with the number of image database."
    
    return img_paths, np.array(features, dtype='object')


def RgbLoader(path):
  """Helper function to read image with PIL.

  Args:
    path: Path to image to be loaded.

  Returns:
    PIL image in RGB format.
  """
  #with tf.io.gfile.GFile(path, 'rb') as f:
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')