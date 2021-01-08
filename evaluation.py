import os
import argparse
import time
import os
import pickle
import numpy as np
from feature_extractor import FeatureExtractor
from utils import load_data, make_index_table, load_features_from_dir, RgbLoader
from config.config import K, pq_config
from retrieval import search
import faiss
import glob2


# Calculate the average precision
def calculate_AP(res_query_lst, true_lst):
    count = 0
    AP = 0
    for (j, e) in enumerate(res_query_lst):
        if e in true_lst:
            count += 1
            AP += count/(j+1)
    if count == 0:
        return 0
    return AP


# perform query for all image and calculate mAP
def calculate_mAP(gt_path, base_dir, db_features, 
                    db_descriptors_np, db_img_from_des,
                    delf_model, pq, k, ransac, query_output):
    
    img_paths = glob2.glob(os.path.join(base_dir, '*.jpg'))[:]
    gt_paths = os.listdir(gt_path)
    gt_paths = [os.path.join(gt_path, file_name) for file_name in gt_paths]
    query_lst, ok_lst, good_lst, junk_lst = [], [], [], []
    for file_path in gt_paths:
        if file_path[-9:] == 'query.txt':
            query_lst.append(file_path)
            ok_lst.append(file_path[:-9] + 'ok.txt')
            good_lst.append(file_path[:-9] + 'good.txt')
            junk_lst.append(file_path[:-9] + 'junk.txt')

    for i, query in enumerate(query_lst):
        with open(query, "r") as f_query:
            query_img_name = f_query.read().split()[0].split("_")[1:]
            query_img_name = '_'.join(query_img_name)
            query_img_path = os.path.join(base_dir, query_img_name) + '.jpg'
            f_query.close()

        with open(ok_lst[i], "r") as f_true:
            true_lst = f_true.read().splitlines()
            f_query.close()
        
        query_img = np.array(RgbLoader(query_img_path))
       
        results = search(img_paths=img_paths, db_features=db_features, 
            db_descriptors_np=db_descriptors_np, db_img_from_des=db_img_from_des,
            delf_model=delf_model, pq=pq,
            query_img=query_img, k=10, ransac=1, 
            query_output='./static/query_output'
        )
        
        results_query_lst = [os.path.basename(res[2]).split(".")[0] for res in results]
        print(results_query_lst)
        input()
        # calculate mAP
        AP.append(calculate_AP(results_query_lst, true_lst))
    return sum(AP)/len(AP)  # mAP


def main(args):
    
    # Read image features
    config_path = 'config/delf_config.pbtxt'
    delf_model = FeatureExtractor(config_path)
    print("[INFO] Loading features ...")
    db_features = load_features_from_dir(path=args['feature_paths'])

    # Stack descriptors of images
    stack_time = time.process_time()
    locations_list = []
    descriptors_list = []
    for db_feature_img in db_features:
        locations, _, descriptors, _, _ = db_feature_img
        if len(descriptors.shape) == 1:
            descriptors = np.array([np.array([1e6]*40)])
            locations = np.array([np.array((-1, -1))])
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
    pq = faiss.read_index(args['pq_path'])
    print("** Build PQ table time: {} (s)".format(time.process_time()-build_pq_time))
    
    calculate_mAP(gt_path=args['gt_path'], base_dir=args['base_dir'], db_features=db_features, 
                    db_descriptors_np=db_descriptors_np, db_img_from_des=db_img_from_des,
                    delf_model=delf_model, pq=pq, k=10, ransac=1, 
                    query_output='./static/query_output',
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate P of model')
    parser.add_argument('--gt_path', type=str, default='./static/database/groundtruth/oxford/', 
                            help='The path to grouth truth path.')
    parser.add_argument('--base_dir', type=str, default='./static/database/images/oxford/', 
                            help='The path to image database.')
    parser.add_argument('--feature_paths', type=str, default='./static/database/features/40D/oxford/', 
                            help='The path to feature files')
    parser.add_argument('--pq_path', type=str, default='./static/PQ/pq_oxford_40D_5063.bin', 
                            help='The path to pq object file')
    args = vars(parser.parse_args())
    main(args)