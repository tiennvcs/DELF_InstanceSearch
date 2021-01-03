import os
import glob2
from delf import feature_io


def load_data(path):
    """
    Return the list of images and list of relative features
    """

    types = ('*.png', '*.jpg') 
    
    features = []
    img_paths = []
    
    base_image_path = os.path.join(path, 'images/')
    base_feature_path = os.path.join(path, 'features/')
    
    for type_img in types:
        img_paths.extend(sorted(glob2.glob(os.path.join(base_image_path, type_img))))
    for feature_path in sorted(glob2.glob(os.path.join(base_feature_path, '*.delf')))[:10]:
        _, _, descriptors, _, _= feature_io.ReadFromFile(feature_path)
        features.append(descriptors)
    
    #assert len(img_paths) == len(features), "The number of features is not campatible with the number of image database."

    return img_paths, features