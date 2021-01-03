import time
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from delf import utils 
from utils import load_data

# Read image features
print("[INFO] Loading database ...")
config_path = 'config/delf_config.pbtxt'
fe = FeatureExtractor(config_path)
img_lst, features = load_data(path='./static/database')


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
        start_time = time.clock()
        query_features = fe.extract(converted_jpg)
        print("Time extract feature: ", time.clock()-start_time)

        # Similarity evaluation


        # Ranking


        scores = []

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run()


