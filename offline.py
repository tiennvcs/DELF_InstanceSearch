from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/database/img").glob("*.png")):
        print(img_path)
        feature = fe.extract(img=Image.open(img_path))

        feature_path = Path("./static/database") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)


