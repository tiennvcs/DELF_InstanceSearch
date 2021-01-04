import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from delf import utils 
from utils import load_data
from config.config import MAX_DESCRIPTOR, K

# Read image features
print("[INFO] Loading database ...")
config_path = 'config/delf_config.pbtxt'
fe = FeatureExtractor(config_path)
img_paths, features = load_data(path='./static/database')


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        converted_jpg = np.array(utils.RgbLoader(uploaded_img_path))

        # Return the local descriptors of query image. with shape [1000, 40] by default
        start_time = time.process_time()
        query_features = fe.extract(converted_jpg)
        print(query_features.keys())
        input()
        print("Time extract feature: ", time.process_time()-start_time)

        # Similarity evaluation
        dists = []
        print(features.shape)
        different_distances = features - np.expand_dims(query_features[:MAX_DESCRIPTOR], 0)
        for distance_descriptor in different_distances:
            distance_descriptor = np.linalg.norm(distance_descriptor, axis=1)
            dists.append(np.linalg.norm(distance_descriptor))

        ids = np.argsort(dists)[:K]
        scores = [(dists[id], img_paths[id]) for id in ids]

        # Ranking


        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run()


