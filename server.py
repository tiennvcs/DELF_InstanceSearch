import time
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from delf import utils 
from utils import load_data, make_index_table
from config.config import K, pq_config
#import sklearn.external.joblib as extjoblib
import h5py
from retrieval import search
import faiss


# Read image features
config_path = 'config/delf_config.pbtxt'
delf_model = FeatureExtractor(config_path)
print("[INFO] Loading database ...")
start_load_time = time.process_time()
img_paths, db_features = load_data(path='./static/database')
print("** Loading time: {} (s)".format(time.process_time()-start_load_time))

# Stack descriptors of images
stack_time = time.process_time()
locations_list = []
descriptors_list = []
for db_feature_img in db_features:
    locations, _, descriptors, _, _ = db_feature_img
    if len(descriptors.shape) == 1:
        continue
    locations_list.append(locations)
    descriptors_list.append(descriptors)

db_descriptors_np = np.concatenate(np.asarray(descriptors_list), axis=0).astype('float32')
print("** Stack time: {} (s)".format(time.process_time()-stack_time))
# End stack

# Index table for database image and feature
index_table_time = time.process_time()
_, db_img_from_des = make_index_table(descriptors_list)
print("** Index table time: {} (s)".format(time.process_time()-index_table_time))
# End index

# Build Product Quantization
build_pq_time = time.process_time()
pq_path = 'static/PQ/pq_40_60.bin'
if not os.path.exists(pq_path):
    #pq = nanopq.PQ(M=pq_config['n_subq'], Ks=pq_config['n_centroids'], verbose=True)
    #pq.fit(vecs=db_descriptors_np, iter=200, seed=18521489)
    dim = 40          # dimension
    n_subq = 8        # number of sub-quantizers
    n_centroids = 64  # number of centroids for each sub-vector
    n_bits = 8        # number of bits for each sub-vector
    n_probe = 4       # number of voronoi cell to explore
    coarse_quantizer = faiss.IndexFlatL2(pq_config['dim'])
    pq = faiss.IndexIVFPQ(coarse_quantizer, pq_config['dim'],
                        pq_config['n_centroids'], pq_config['n_subq'], pq_config['n_bits']) 
    pq.nprobe = pq_config['n_probe']
    pq.train(db_descriptors_np)
    pq.add(db_descriptors_np)
    faiss.write_index(pq, pq_path)
pq = faiss.read_index(pq_path)  # index2 is identical to index
print("** Build PQ table time: {} (s)".format(time.process_time()-build_pq_time))
# End build


# Start web service
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        load_time = time.process_time()
        converted_jpg = np.array(utils.RgbLoader(uploaded_img_path))
        print("** Load query image time: ", time.process_time()-load_time)

        # Search 
        s_time = time.process_time()
        scores = search(img_paths=img_paths, db_features=db_features, 
            db_descriptors_np=db_descriptors_np, db_img_from_des=db_img_from_des,
            delf_model=delf_model, pq=pq, #descriptors_code=descriptors_code,
            query_img=converted_jpg, k=10, ransac=1, 
            query_output='./static/query_output'
        )
        print("** Search time: ", time.process_time()-s_time)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run()


