import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import retrieve


def get_bag_of_words_dir():
    result = Path(__file__).parent / 'data' / 'bag_of_words'
    result.mkdir(parents=True, exist_ok=True)
    return result


def get_dataset(data_dir):
    datapath = data_dir / 'bag_of_words.data.npz'
    if datapath.exists():
        with open(str(datapath), 'rb') as ifh:
            data = np.load(ifh, allow_pickle=True)
            X = data['X']
            if X.shape == ():
                # scipy sparse matrix
                X = X[()]
            return X, data['y']
    bow_data = build_bag_of_words()
    X = bow_data['bag_of_words']
    y = bow_data['ratings']
    with open(str(datapath), 'wb') as ofh:
        np.savez(ofh, X=X, y=y)
    return X, y


class SnippetVectorizer:

    def __init__(self, snippet_pairs):
        self._snip2ind = {}
        snippets = []
        for source_snip, target_snip in snippet_pairs:
            if source_snip not in self._snip2ind:
                self._snip2ind[source_snip] = len(self._snip2ind)
                snippets.append(source_snip)
            if target_snip not in self._snip2ind:
                self._snip2ind[target_snip] = len(self._snip2ind)
                snippets.append(target_snip)
        vectorizer = CountVectorizer()
        self._matrix = vectorizer.fit_transform(snippets)
        self._vocab = vectorizer.get_feature_names()

    def get_word_counts(self, snippet):
        if snippet not in self._snip2ind:
            raise ValueError(f'Unknown snippet: {snippet}')
        snipind = self._snip2ind[snippet]
        data = self._matrix.data
        indices = self._matrix.indices
        indptr = self._matrix.indptr
        return {
            a: b
            for a, b in zip(indices[indptr[snipind]:indptr[snipind + 1]],
                            data[indptr[snipind]:indptr[snipind + 1]])
        }

    def get_vocab_size(self):
        return len(self._vocab)

    def save_vocab(self, outpath):
        with open(str(outpath), 'w') as ofh:
            for word in self._vocab:
                ofh.write(word)
                ofh.write('\n')


def build_bag_of_words():
    bow_path = get_bag_of_words_dir() / 'bag_of_words.data.pickle'
    if bow_path.exists():
        with bow_path.open('rb') as ifh:
            return pickle.load(ifh)
    raw_keys, _, v5_snippet_pairs, ratings = retrieve.extract_benchmark_data()
    snipvecs = SnippetVectorizer(v5_snippet_pairs)
    vocab_size = snipvecs.get_vocab_size()
    data = []
    row_ind = []
    col_ind = []
    for v5_snippet_pair in v5_snippet_pairs:
        source_snippet, target_snippet = v5_snippet_pair
        cur_row = 0 if len(row_ind) == 0 else row_ind[-1] + 1
        source_counts = snipvecs.get_word_counts(source_snippet)
        target_counts = snipvecs.get_word_counts(target_snippet)
        row_ind.extend([cur_row] * (len(source_counts) + len(target_counts)))
        for word, count in source_counts.items():
            col_ind.append(word)
            data.append(count)
        for word, count in target_counts.items():
            col_ind.append(word + vocab_size)
            data.append(count)
    shape = (len(v5_snippet_pairs), 2 * vocab_size)
    results = {
        'raw_keys': raw_keys,
        'bag_of_words': csr_matrix((data, (row_ind, col_ind)), shape=shape),
        'ratings': np.array(ratings)
    }
    with bow_path.open('wb') as ofh:
        pickle.dump(results, ofh)
    return results
