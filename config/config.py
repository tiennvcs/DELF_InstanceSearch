import faiss


MAX_DESCRIPTOR = 500
K = 10


pq_config = {
    'dim': 40,          # dimension
    'n_subq': 8,        # number of sub-quantizers
    'n_centroids': 32,  # number of centroids for each sub-vector
    'n_bits': 5,        # number of bits for each sub-vector
    'n_probe': 3,       # number of voronoi cell to explore
}