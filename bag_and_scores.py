import numpy as np
from scipy.sparse import csr_matrix

import learning


def get_dataset(data_dir):
    datapath = data_dir / 'bag_and_scores.data.npz'
    if datapath.exists():
        with open(str(datapath), 'rb') as ifh:
            data = np.load(ifh, allow_pickle=True)
            X = data['X']
            if X.shape == ():
                # scipy sparse matrix
                X = X[()]
            return X, data['y']
    import bag_of_words
    bow_data = bag_of_words.build_bag_of_words()
    results = learning.get_filtered_results(data_dir)
    X, y = build_dataset(results, bow_data)
    with open(str(datapath), 'wb') as ofh:
        np.savez(ofh, X=X, y=y)
    return X, y


def build_dataset(results, bow_data):
    SCORE_OFFSET = 3
    raw_keys = bow_data['raw_keys']
    bag_of_words = bow_data['bag_of_words']
    assert bag_of_words.has_sorted_indices
    offset_js = bag_of_words.indices + SCORE_OFFSET
    indptr = bag_of_words.indptr
    orig_data = bag_of_words.data
    row_ind = []
    col_ind = []
    data = []
    tess_intervals = learning.build_interval_tree(results)
    for i, raw_key in enumerate(raw_keys):
        if raw_key in results:
            locus_key = raw_key
        else:
            locus_key = learning.interval_lookup(tess_intervals, raw_key)
        if locus_key in results and \
                len(results[locus_key]) == 1:
            snippet_key = next(iter(results[locus_key].keys()))
            scores = results[locus_key][snippet_key]
        else:
            scores = {}
        for j, score_type in enumerate(['lemmata', 'semantic', 'sem_lem']):
            row_ind.append(i)
            col_ind.append(j)
            if score_type in scores:
                data.append(scores[score_type])
            else:
                data.append(np.nan)
        start = indptr[i]
        stop = indptr[i + 1]
        row_ind.extend([i] * (stop - start))
        col_ind.extend(offset_js[start:stop])
        data.extend(orig_data[start:stop])
    orig_shape = bag_of_words.shape
    return csr_matrix(
        (data, (row_ind, col_ind)),
        shape=(orig_shape[0],
               orig_shape[1] + SCORE_OFFSET)), bow_data['ratings']
