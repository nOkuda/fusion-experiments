import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.preprocessing
from intervaltree import IntervalTree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

import retrieve
import snippet


def experiment(X, y, models, identifier, data_dir):
    data_dir = data_dir / identifier
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    resultspath = data_dir / f'results.{identifier}.txt'
    with open(str(resultspath), 'w') as ofh:
        for modelname, model in models:
            print(modelname)
            conf_mat = get_confusion_matrix(data_dir, identifier, modelname,
                                            model, X, y)
            categories = [str(a) for a in range(1, 6)]
            save_confusion_matrix(conf_mat, categories, data_dir, identifier,
                                  modelname)
            accuracy = compute_accuracy(conf_mat)
            mcc = compute_matthews_corrcoef(conf_mat)
            f1_score = compute_f1(conf_mat)
            recall_5 = compute_recall_5(conf_mat)
            precision_5 = compute_precision_5(conf_mat)
            f1_5 = compute_f1_5(conf_mat)
            recall_meaningful = compute_recall_meaningful(conf_mat)
            precision_meaningful = compute_precision_meaningful(conf_mat)
            f1_meaningful = compute_f1_meaningful(conf_mat)
            mcc_5 = compute_mcc_5(conf_mat)
            mcc_meaningful = compute_mcc_meaningful(conf_mat)
            mae = compute_mae(conf_mat)
            weighted_mae = compute_weighted_mae(conf_mat)
            ofh.write(
                f'{modelname}\t{accuracy}\t{mcc}\t{f1_score}\t'
                f'{recall_5}\t{precision_5}\t{f1_5}\t{recall_meaningful}\t'
                f'{precision_meaningful}\t{f1_meaningful}\t'
                f'{mcc_5}\t{mcc_meaningful}\t{mae}\t{weighted_mae}\n')


def get_confusion_matrix(data_dir, identifier, modelname, model, X, y):
    confmatfilepath = data_dir / f'raw.{identifier}.confusion.{modelname}.npy'
    if confmatfilepath.exists():
        with open(str(confmatfilepath), 'rb') as ifh:
            return np.load(ifh)
    predicted = cross_val_predict(model, X, y)
    conf_mat = confusion_matrix(y, predicted)
    with open(str(confmatfilepath), 'wb') as ofh:
        np.save(ofh, conf_mat)
    return conf_mat


def save_confusion_matrix(conf_mat, categories, data_dir, identifier,
                          modelname):
    # plotting with axis=1 tells me what the model learned;
    normalized_conf_mat = sklearn.preprocessing.normalize(conf_mat,
                                                          axis=1,
                                                          norm='l1')
    filename = f'learnplot.{identifier}.confusion.{modelname}.svg'
    outplotpath = data_dir / filename
    plot_confusion_matrix(conf_mat, normalized_conf_mat, categories,
                          outplotpath)


def plot_confusion_matrix(conf_mat, normalized_conf_mat, categories,
                          outplotpath):
    fig, ax = plt.subplots()
    ax.imshow(normalized_conf_mat)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="w")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('prediction')
    ax.set_ylabel('benchmark label')
    fig.tight_layout()
    plt.savefig(str(outplotpath))
    fig.clear
    plt.close(fig)


def compute_accuracy(conf_mat):
    return conf_mat.diagonal().sum() / conf_mat.sum()


def compute_matthews_corrcoef(conf_mat):
    y_true, y_pred = _extract_true_and_pred(conf_mat)
    return sklearn.metrics.matthews_corrcoef(y_true, y_pred)


def _extract_true_and_pred(conf_mat):
    y_true = []
    y_pred = []
    for i, row in enumerate(conf_mat):
        for j, count in enumerate(row):
            y_true.extend([i] * count)
            y_pred.extend([j] * count)
    return np.array(y_true), np.array(y_pred)


def compute_f1(conf_mat):
    y_true, y_pred = _extract_true_and_pred(conf_mat)
    return sklearn.metrics.f1_score(y_true, y_pred, average='macro')


def compute_recall_5(conf_mat):
    row_5 = conf_mat[-1, :]
    if row_5[-1] == 0:
        return 0.0
    return row_5[-1] / row_5.sum()


def compute_precision_5(conf_mat):
    col_5 = conf_mat[:, -1]
    if col_5[-1] == 0:
        return 0.0
    return col_5[-1] / col_5.sum()


def compute_f1_5(conf_mat):
    precision = compute_precision_5(conf_mat)
    recall = compute_recall_5(conf_mat)
    if precision == 0 and recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_mcc_5(conf_mat):
    y_true = []
    y_pred = []
    for i, row in enumerate(conf_mat):
        true_label = 1 if i >= 4 else 0
        for j, count in enumerate(row):
            pred_label = 1 if j >= 4 else 0
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)
    return sklearn.metrics.matthews_corrcoef(y_true, y_pred)


def compute_recall_meaningful(conf_mat):
    meaningful_rows = conf_mat[2:, :]
    captured = meaningful_rows[:, 2:].sum()
    if captured == 0:
        return 0.0
    return captured / meaningful_rows.sum()


def compute_precision_meaningful(conf_mat):
    meaningful_cols = conf_mat[:, 2:]
    captured = meaningful_cols[2:, :].sum()
    if captured == 0:
        return 0.0
    return captured / meaningful_cols.sum()


def compute_f1_meaningful(conf_mat):
    precision = compute_precision_meaningful(conf_mat)
    recall = compute_recall_meaningful(conf_mat)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_mcc_meaningful(conf_mat):
    y_true = []
    y_pred = []
    for i, row in enumerate(conf_mat):
        true_label = 1 if i >= 2 else 0
        for j, count in enumerate(row):
            pred_label = 1 if j >= 2 else 0
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)
    return sklearn.metrics.matthews_corrcoef(y_true, y_pred)


def compute_mae(conf_mat):
    y_true, y_pred = _extract_true_and_pred(conf_mat)
    return sklearn.metrics.mean_absolute_error(y_true, y_pred)


def compute_weighted_mae(conf_mat):
    y_true, y_pred = _extract_true_and_pred(conf_mat)
    y_counts = np.bincount(y_true)
    abs_errors = np.abs(y_true - y_pred)
    # https://stats.stackexchange.com/a/375956 noted that Baccianella et al.
    # 2009 Evaluation Measures for Ordinal Regression recommended macroaverage
    # MAE
    result = 0.0
    for i, count in enumerate(y_counts):
        relevant_abs_errors = abs_errors[y_true == i]
        cur_class_mae = np.sum(relevant_abs_errors) / count
        result += cur_class_mae / len(y_counts)
    return result


def get_filtered_results(data_dir):
    results = {}
    data = retrieve.get_data(data_dir)
    for feature, tess_out in data.items():
        for parallel in tess_out['parallels']:
            lucan_locus = parallel['target_tag'].split()[-1]
            if lucan_locus.startswith('1.'):
                vergil_locus = parallel['source_tag'].split()[-1]
                locus_key = (vergil_locus, lucan_locus)
                snippet_key = (parallel['source_snippet'],
                               parallel['target_snippet'])
                if locus_key not in results:
                    results[locus_key] = {}
                if snippet_key not in results[locus_key]:
                    results[locus_key][snippet_key] = {}
                results[locus_key][snippet_key][feature] = parallel['score']
    return results


def build_interval_tree(data):
    """Construct an interval tree based on data

    data should have the following format:
    {
        (source_locus, target_locus): {
            (source_snippet, target_snippet): {
                feature: score
            }
        }
    }

    resulting data structure is actually a dict -> interval tree -> dict ->
    interval tree, where the first dict indexes by source book, the first
    interval tree indexes by source line, the second dict indexes by target
    book, and the second interval tree indexes by target line:
    {
        source_book: source_line_intervals -> {
            target_book: target_line_intervals
        }
    }
    """
    result = {}
    for (source_locus, target_locus), snippet_level in data.items():
        # TODO this will utterly break when line numbering gets complicated
        # e.g., 493a, 493b
        source_book_str, source_line_str = source_locus.split('.')
        target_book_str, target_line_str = target_locus.split('.')
        source_lines_start = int(source_line_str)
        target_lines_start = int(target_line_str)
        for (source_snippet, target_snippet) in snippet_level.keys():
            source_lines_end = source_lines_start + snippet.count_lines_length(
                source_snippet)
            target_lines_end = target_lines_start + snippet.count_lines_length(
                target_snippet)
            if source_book_str not in result:
                result[source_book_str] = IntervalTree()
            source_tree = result[source_book_str]
            if not source_tree.overlaps(source_lines_start, source_lines_end):
                source_tree[source_lines_start:source_lines_end] = {}
            for interval in source_tree[source_lines_start:source_lines_end]:
                if interval.begin == source_lines_start and \
                        interval.end == source_lines_end:
                    target_dict = interval.data
            assert 'target_dict' in locals()
            if target_book_str not in target_dict:
                target_dict[target_book_str] = IntervalTree()
            target_tree = target_dict[target_book_str]
            target_tree[target_lines_start:target_lines_end] = True
    return result


def interval_lookup(interval_tree, tags):
    """Look up tags in interval_tree

    If tags cannot be found in interval_tree, return tags unchanged
    """
    source_locus, target_locus = tags
    source_book_str, source_line_str = source_locus.split('.')
    target_book_str, target_line_str = target_locus.split('.')
    source_line = int(source_line_str)
    target_line = int(target_line_str)
    if source_book_str in interval_tree:
        source_tree = interval_tree[source_book_str]
        for source_interval in sorted(source_tree[source_line]):
            target_dict = source_interval.data
            if target_book_str in target_dict:
                target_tree = target_dict[target_book_str]
                for target_interval in sorted(target_tree[target_line]):
                    return (f'{source_book_str}.{source_interval.begin}',
                            f'{target_book_str}.{target_interval.begin}')
    return tags


def get_locus_key(raw_key, results, interval_tree):
    """Translate the source and target loci in raw_key for lookup in results

    If the loci are already in results, then no translation is necessary.

    If the loci are not already in results, then attempts to translate the loci
    specified in raw_key into loci that correspond to something in results.

    If no translation is possible, then returns the raw_key as is.
    """
    if raw_key in results:
        return raw_key
    return interval_lookup(interval_tree, raw_key)
