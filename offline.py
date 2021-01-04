import os
import numpy as np
from PIL import Image
from pathlib import Path
from utils import RgbLoader
from delf import feature_io
from feature_extractor import FeatureExtractor
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

_DELF_EXT = '.delf'

if __name__ == '__main__':

    config_path = 'config/delf_config.pbtxt'
    output_path = './static/database/features/'

    fe = FeatureExtractor(config_path)

    for img_path in sorted(Path("./static/database/images").glob("*.jpg")):

        out_desc_fullpath = os.path.join(output_path, os.path.split(img_path)[-1].split(".")[0]+_DELF_EXT)
        if os.path.exists(out_desc_fullpath):
            print("Skip {}".format(img_path))
            continue
        img = np.array(RgbLoader(img_path))
        extracted_features = fe.extract(img)
        locations_out = extracted_features['local_features']['locations']
        descriptors_out = extracted_features['local_features']['descriptors']
        feature_scales_out = extracted_features['local_features']['scales']
        attention_out = extracted_features['local_features']['attention']
        feature_io.WriteToFile(out_desc_fullpath, 
                                locations_out, 
                                feature_scales_out,
                                descriptors_out, 
                                attention_out)
        print("Extracted feature from image {}".format(img_path))



