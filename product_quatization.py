import os
import glob2
import nanopq
import pickle
import argparse
import numpy as np
from delf import feature_io


def product_quantization(vectors:np.ndarray, M:int, k:int, output_path):
    
    # Check the second dimension of vectors divisible for M
    assert vectors.shape[-1] % M == 0, "The dimension of vector is NOT DIVISIBLE for M"

    # Instantiate with M=8 sub-spaces    
    pq = nanopq.PQ(M=M, Ks=k, verbose=True)

    # Train codewords
    pq.fit(vectors, iter=100, seed=18521489)

    # Save the QP-codes to disk
    save_file = os.path.join(output_path, 'pq_{}_{}'.format(M, k) + '.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(pq, f)
    
    print("Saving the PQ to {}".format(save_file))

    return pq


def main(args):

    path = './static/database/'
    types = ('*.png', '*.jpg')
    
    features = []
    img_paths = []
    
    base_image_path = os.path.join(path, 'images/')
    base_feature_path = os.path.join(path, 'features/')
    
    for type_img in types:
        img_paths.extend(sorted(glob2.glob(os.path.join(base_image_path, type_img))))
    
    for i, feature_path in enumerate(sorted(glob2.glob(os.path.join(base_feature_path, '*.delf')))):
        _, _, descriptors, _, _ = feature_io.ReadFromFile(feature_path)
        features.extend(descriptors)
        #img_features_dict[i]

    features = np.array(features, dtype=np.float32)
    print(features.shape)
    pq = product_quantization(vectors=features, M=10, k=60, output_path='./static/PQ/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Product quantization')
    parser.add_argument('-M', default=10, required=True, type=int,
                        help='The number of subcodebooks (subspaces)')
    parser.add_argument('-k', default=40, required=True, type=int,
                        help='The number of iterations in K-mean algorithm')
    parser.add_argument('--output_path', default='./static/PQ/', 
                        required=True, help='The path to save QP-codes')
    args = vars(parser.parse_args())
    print(args)

    main(args)
