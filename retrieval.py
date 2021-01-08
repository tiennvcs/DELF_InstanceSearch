import os
import time
import faiss
import pickle
#import nanopq
import argparse
from scipy import spatial
import numpy as np
from utils import load_data, RgbLoader
from feature_extractor import FeatureExtractor
from config.config import pq_config
from shutil import copyfile

from skimage import feature
from skimage import measure
from skimage import transform

_DISTANCE_THRESHOLD = 0.8


def reranking(img_paths, features, query_img, query_feature):
    
    query_descriptors = query_feature['descriptors']
    query_locations = query_feature['locations']
    num_query_features = query_locations.shape[0]
    query_tree = spatial.cKDTree(query_descriptors)

    scores = []

    for i, img_path in enumerate(img_paths):
        locations, _, descriptors, _, _ = features[i]
        num_features = locations.shape[0]
        
        # Find nearest-neighbor matches using a KD tree.
        _, indices = query_tree.query(
            descriptors, distance_upper_bound=_DISTANCE_THRESHOLD)

        # Select feature locations for putative matches.
        locations_2_to_use = np.array([
            locations[i,] 
            for i in range(num_features) 
            if indices[i] != num_query_features]
        )
        
        locations_1_to_use = np.array([
            query_locations[indices[i],]
            for i in range(num_features)
            if indices[i] != num_query_features]
        )

        # Perform geometric verification using RANSAC.
        Model, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                    transform.AffineTransform,
                                    min_samples=3,
                                    residual_threshold=20,
                                    max_trials=1000)
        #print(f'Found {sum(inliers)} inliers')

        scores.append((sum(inliers), Model, img_paths[i]))
    
    scores.sort(key=lambda x: -x[0])
    # for (inliers,_,idx_path) in scores:
    #     print(inliers,img_paths[idx_path])

    return scores


def search(img_paths, db_features, db_descriptors_np, 
            delf_model, query_img, pq, #descriptors_code,
            k, ransac, query_output, db_img_from_des):


    # Extract feature from image
    start_time_extraction = time.process_time()
    query_features = delf_model.extract(img=query_img)
    print("**** Extract time: {} (s)".format(time.process_time()-start_time_extraction))
    query_descriptors = query_features['descriptors']
    
    start_time_search_pq = time.process_time()
    #results = []
    # for descriptor in query_descriptors:
    #     dt = pq.dtable(query=descriptor).adist(descriptors_code)
    #     indices = np.argsort(dt)[:k]
    #     results.append(indices)
    _, query_des2desList = pq.search(query_descriptors, k)
    print("**** PQ time: {} (s)".format(time.process_time()-start_time_search_pq))    

    start_time_inverted = time.process_time()
    query_des2desList = np.array(query_des2desList)
    img_frequencies = dict()
    for descriptor_id_lst in query_des2desList:
        for descriptor_id in descriptor_id_lst:
            if not db_img_from_des[descriptor_id] in img_frequencies.keys():
                img_frequencies[db_img_from_des[descriptor_id]] = 1
            else:
                img_frequencies[db_img_from_des[descriptor_id]] += 1
    sorted_img_frequencies = sorted(img_frequencies.items(), key=lambda x: x[1], reverse=True)
    selected_top_imgs = [value[0] for value in sorted_img_frequencies[:k]]
    print("**** Reverted time: {} (s)".format(time.process_time()-start_time_inverted))

    if ransac == 0:
        scores = [(
            img_frequencies[img_id], 0, img_paths[img_id]) 
            for img_id in selected_top_imgs
        ]
        return scores
    
    start_ransac_time = time.process_time()
    scores = reranking(
        img_paths=np.array(img_paths)[selected_top_imgs],
        features=db_features[selected_top_imgs],
        query_img=query_img,
        query_feature=query_features,
    )
    print("**** Reranking time: {} (s)".format(time.process_time()-start_ransac_time))
    # scores = reranking(
    #     img_paths=np.array(img_paths)[:],
    #     features=db_features[:],
    #     query_img=query_img,
    #     query_feature=query_features,
    # )
    return scores


def main(args):

    print(args)

    # 1. Load all features into disk
    print("[INFO] Loading database images and features ...")
    img_paths, db_features = load_data(path=args['db_path'])
    
    print("--> The number of images are {}".format(len(img_paths)))

    # Load DELF model
    print("[INFO] Initilize DELF mode by {} config ...".format(args['model_config']))
    delf_model = FeatureExtractor(args['model_config'])

    # Load query image
    query_img = np.array(RgbLoader(args['query_path']))
    output_path = os.path.join('static/output_query/', os.path.split(args['query_path'])[-1].split(".")[0])

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # perform search
    top_img_list = search(img_paths, db_features, delf_model, 
                            query_img, args['k'], args['ransac'], args['pq_path'], output_path)

    # Visualize results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform retrieval image')
    parser.add_argument('--db_path', type=str, default='./static/database/', 
                            help='The path to database container images and extracted features.')
    parser.add_argument('--model_config', type=str, 
                            default='./config/delf_config.pbtxt',
                            help='The path to config DELF model to use.')
    parser.add_argument('--query_path', type=str, 
                            default='./static/query/image_1.jpg',
                            help='The path to query image.')
    parser.add_argument('--pq_path', type=str, 
                            default='./static/PQ/pq_40_60.pkl',
                            help='The path to product quantization table.')
    parser.add_argument('--ransac', type=int, default=0, 
                        help='Use ransac algorithm to re-ranking results or not')
    parser.add_argument('-k', type=int, default=10, 
                        help='The top k results from retrieval')

    args = vars(parser.parse_args())

    main(args)



    