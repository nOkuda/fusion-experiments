from pathlib import Path

import experiment
from plot import make_plots


def _main():
    print('# Running experiments')
    params = [
        Parameters(
            name='meaningful',
            categories=['meaningless', 'meaningful'],
            transformer=meaningful_transformer,
        ),
        Parameters(
            name='fives',
            categories=['non-5', '5'],
            transformer=fives_transformer,
        ),
        Parameters(
            name='final',
            categories=[str(a) for a in range(1, 6)],
            transformer=shift_transformer,
        ),
    ]
    data_dir = Path(__file__).parent / 'data'
    for cur_params in params:
        exp_dir = data_dir / cur_params.name
        experiment.run(exp_dir, cur_params.transformer, cur_params.categories)
    print('# Writing out results')
    for exp_type in ['meaningful', 'fives', 'final']:
        write_results(exp_type)
    print('# Making plots')
    make_plots()


class Parameters:

    def __init__(self, name, categories, transformer):
        self.name = name
        self.categories = categories
        self.transformer = transformer


def meaningful_transformer(y):
    result = y.copy()
    result[result < 3] = 0
    result[result >= 3] = 1
    return result


def fives_transformer(y):
    result = y.copy()
    result[result < 5] = 0
    result[result >= 5] = 1
    return result


def shift_transformer(y):
    result = y.copy()
    result -= 1
    return result


def write_results(exp_type):
    resultsdir = Path(__file__).parent / 'results'
    if not resultsdir.exists():
        resultsdir.mkdir(parents=True, exist_ok=True)
    outfilepath = resultsdir / f'{exp_type}_results.txt'
    with outfilepath.open('w') as ofh:
        ofh.write('Input\tModel\tImp.\tBal.\tNorm.\tAcc.\tF1\tMCC\n')
        features = [
            ('scores_only', 'all scores'),
            ('bag_of_words', 'words'),
            ('bag_and_scores', 'words and scores'),
            ('lemmata_only', 'lemmata only'),
            ('semantic_only', 'synonyms only'),
            ('lem+sem', 'lemmata and synonyms'),
            ('lem+sem_lem', 'lemmata and synonyms + lemmata'),
            ('sem_lem_only', 'synonyms + lemmata only'),
            ('sem+sem_lem', 'synonyms and synonyms + lemmata'),
        ]
        for features_used, label in features:
            data_dir = Path(
                __file__).parent / 'data' / exp_type / features_used
            results_path = data_dir / f'results.{features_used}.txt'
            for results in _read_results(results_path):
                out_results = build_out_results(label, results)
                ofh.write('\t'.join(out_results))
                ofh.write('\n')


def _read_results(result_path):
    with result_path.open() as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                items = line.split('\t')
                yield tuple(items)


def build_out_results(label, results):
    model_name = results[0]
    balanced = 'no'
    if 'oversampled' in model_name or 'weighted' in model_name:
        balanced = 'yes'
    return (label, 'NN' if 'neural' in model_name else 'MLR',
            'mean' if 'imputed' in model_name else '0', balanced,
            'yes' if 'normalized' in model_name else 'no', *results[1:])


if __name__ == '__main__':
    _main()
