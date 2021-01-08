import faiss


#MAX_DESCRIPTOR = 1000
K = 10


pq_config = {
    'dim': 40,          # dimension
    'n_subq': 8,        # number of sub-quantizers
    'n_centroids': 64,  # number of centroids for each sub-vector
    'n_bits': 8,        # number of bits for each sub-vector
    'n_probe': 8,       # number of voronoi cell to explore
}