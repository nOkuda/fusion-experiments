import numpy as np

import learning
import retrieve


def get_dataset(data_dir):
    datapath = data_dir / 'scores_only.data.npz'
    if datapath.exists():
        with open(str(datapath), 'rb') as ifh:
            data = np.load(ifh)
            return data['X'], data['y']
    results = learning.get_filtered_results(data_dir)
    benchmark = retrieve.get_benchmark_data()
    X, y = build_dataset(results, benchmark)
    with open(str(datapath), 'wb') as ofh:
        np.savez(ofh, X=X, y=y)
    return X, y


def build_dataset(results, benchmark):
    X = []
    y = []
    tess_intervals = learning.build_interval_tree(results)
    for raw_key, values in benchmark.items():
        if raw_key in results:
            locus_key = raw_key
        else:
            locus_key = learning.interval_lookup(tess_intervals, raw_key)
        if locus_key in results and \
                len(values) == 1 and \
                len(results[locus_key]) == 1:
            category = values[0][1]
            snippet_key = next(iter(results[locus_key].keys()))
            scores = results[locus_key][snippet_key]
            X.append([
                scores['lemmata'] if 'lemmata' in scores else np.nan,
                scores['semantic'] if 'semantic' in scores else np.nan,
                scores['sem_lem'] if 'sem_lem' in scores else np.nan,
            ])
            y.append(category)
        else:
            assert len(values) == 1, len(values)
            category = values[0][1]
            X.append([np.nan, np.nan, np.nan])
            y.append(category)
    return np.array(X), np.array(y)
