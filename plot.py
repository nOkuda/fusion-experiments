from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing

import scores_only
from learning import plot_confusion_matrix


def _main():
    make_plots()


def make_plots():
    outdirpath = Path(__file__).parent / 'plots'
    if not outdirpath.exists():
        outdirpath.mkdir(parents=True, exist_ok=True)
    plot_perfect_confusion(outdirpath)
    plot_parallels_distribution(outdirpath)
    make_separability_plots(outdirpath)


def plot_perfect_confusion(outdirpath):
    conf_mat = np.array([[490, 0, 0, 0, 0], [0, 2284, 0, 0, 0],
                         [0, 0, 423, 0, 0], [0, 0, 0, 113, 0],
                         [0, 0, 0, 0, 100]])
    normalized_conf_mat = sklearn.preprocessing.normalize(conf_mat,
                                                          axis=1,
                                                          norm='l1')
    outfilepath = outdirpath / 'perfect_confusion.svg'
    plot_confusion_matrix(conf_mat, normalized_conf_mat,
                          ['1', '2', '3', '4', '5'], str(outfilepath))


def plot_parallels_distribution(outdirpath):
    data_dir = Path(__file__).parent / 'data' / 'phrase'
    X, y = scores_only.get_dataset(data_dir)
    print(X.shape)
    X[np.isnan(X)] = 0
    X[X != 0] = 1
    associated_scores_count = X.sum(axis=1)
    all_scores = associated_scores_count >= 3
    all_values = y
    some_values = y[associated_scores_count == 0]
    title = 'Distribution of Ratings'
    fig, ax = plt.subplots()
    data = {
        'All parallels': all_values,
        'Parallels with no scores': some_values
    }
    for label, values in data.items():
        bins = np.bincount(values)
        counter = {a: b for a, b in enumerate(bins) if b > 0}
        plot_axbar(ax, counter, title, label=label)
    plt.legend()
    fig.tight_layout()
    outfilepath = outdirpath / 'no_scores_dist.svg'
    fig.savefig(str(outfilepath))
    fig.clear
    plt.close(fig)

    all_scores_count = np.sum(all_scores)
    print(all_scores_count)
    print(X.shape[0] - all_scores_count)
    print(np.sum(associated_scores_count == 0))


def plot_axbar(ax, counter, title, ylim=None, label=None):
    categories = [a for a in sorted(counter.keys())]
    values = [counter[c] for c in categories]
    x_pos = [i for i in range(len(categories))]
    rects = ax.bar(x_pos, values, label=label)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            '{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)


def make_separability_plots(outdirpath):
    plot_ideal_separability(outdirpath)
    data_dir = Path(__file__).parent / 'data' / 'phrase'
    X, y = scores_only.get_dataset(data_dir)
    fives = np.array(['5' if a == 5 else 'non-5' for a in y])
    outfilepath = outdirpath / 'separability_fives.svg'
    plot_separability(X, fives, ['non-5', '5'], str(outfilepath))
    meaningfuls = np.array(
        ['meaningful' if a >= 3 else 'meaningless' for a in y])
    outfilepath = outdirpath / 'separability_meaningfuls.svg'
    plot_separability(X, meaningfuls, ['meaningless', 'meaningful'],
                      str(outfilepath))


def plot_ideal_separability(outdirpath):
    from numpy.random import default_rng
    rng = default_rng()
    mean = [3, 3, 3]
    cov = [[1, 0.5, 0.2], [0.5, 2, 1], [0.2, 1, 1]]
    cat1 = rng.multivariate_normal(mean, cov, size=3000)
    mean = [10, 7, 9]
    cov = [[1, 0.5, 0.1], [0.5, 1, 0.7], [0.1, 0.7, 1]]
    cat2 = rng.multivariate_normal(mean, cov, size=100)
    X = np.concatenate((cat1, cat2), axis=0)
    y = np.concatenate((['A'] * 3000, ['B'] * 100), axis=0)
    outfilepath = outdirpath / 'separability_ideal.svg'
    plot_separability(X, y, ['A', 'B'], str(outfilepath))


def plot_separability(X, y, hue_order, outfilename):
    data = pd.DataFrame(X,
                        index=[a for a in range(X.shape[0])],
                        columns=['lemmata', 'synonyms', 'synonyms + lemmata'])
    data['rating'] = y
    sns.pairplot(data, hue='rating', hue_order=hue_order, kind='kde')
    plt.savefig(outfilename)


if __name__ == '__main__':
    _main()
