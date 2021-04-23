"""Retrieve phrases"""
import gzip
import json
from pathlib import Path

import requests
from fuzzywuzzy import process

import search
from snippet import PhrasesTracker

NAME_CONVERSION = {'aeneid': 'aeneid', 'bellum civile': 'lucan'}


def get_benchmark_data():
    results = {}
    benchmark_path = Path(__file__).parent / 'data' / 'aen_luc1_hand.txt'
    with benchmark_path.open() as ifh:
        # skip header line
        next(ifh)
        for line in ifh:
            line = line.strip()
            if line:
                data = line.split('\t')
                lucan_locus = f'{data[0]}.{data[1]}'
                lucan_quote = data[2]
                vergil_locus = f'{data[3]}.{data[4]}'
                vergil_quote = data[5]
                rank = int(data[6])
                key = (vergil_locus, lucan_locus)
                if key in results:
                    results[key].append(((vergil_quote, lucan_quote), rank))
                else:
                    results[key] = [((vergil_quote, lucan_quote), rank)]
    return results


def extract_benchmark_data():
    benchmark = get_benchmark_data()
    phrases_finder = build_phrases_finder()
    raw_keys = []
    benchmark_snippet_pairs = []
    v5_snippet_pairs = []
    ratings = []
    for raw_key, values in benchmark.items():
        raw_keys.append(raw_key)
        benchmark_snippet_pairs.append(values[0][0])
        v5_snippet_pairs.append(
            phrases_finder.find(raw_key, benchmark_snippet_pairs[-1]))
        ratings.append(values[0][1])
    return raw_keys, benchmark_snippet_pairs, v5_snippet_pairs, ratings


class PhrasesFinder:

    def __init__(self):
        self.source_intervals = PhrasesTracker(
            retrieve_phrases('vergil', 'aeneid')['units'])
        self.target_intervals = PhrasesTracker(
            retrieve_phrases('lucan', 'bellum civile')['units'])

    def find(self, raw_key, benchmark_snippets):
        source_snippets = self.source_intervals.find(raw_key[0])
        if len(source_snippets) == 1:
            source_snippet = source_snippets[0]
        else:
            source_snippet = process.extractOne(benchmark_snippets[0],
                                                source_snippets)[0]
        target_snippets = self.target_intervals.find(raw_key[1])
        if len(target_snippets) == 1:
            target_snippet = target_snippets[0]
        else:
            target_snippet = process.extractOne(benchmark_snippets[1],
                                                target_snippets)[0]
        return source_snippet, target_snippet


def build_phrases_finder():
    return PhrasesFinder()


def _main():
    for author, title in [('vergil', 'aeneid'), ('lucan', 'bellum civile')]:
        retrieve_phrases(author, title)


def retrieve_phrases(author, title):
    json_filename = f'{NAME_CONVERSION[title]}_phrases.json'
    outfilepath = Path(__file__).parent / 'data' / json_filename
    if not outfilepath.exists():
        work_id = search._retrieve_work_id(author, title)
        r = requests.get(f'{search.API_URL}units/',
                         params={
                             'works': work_id,
                             'unit_type': 'phrase'
                         },
                         allow_redirects=False)
        phrases = r.json()
        with outfilepath.open('w') as ofh:
            json.dump(phrases, ofh)
        return phrases
    with outfilepath.open() as ifh:
        return json.load(ifh)


def get_data(data_dir):
    results_files = [
        data_dir / 'lemmata.json.gz',
        data_dir / 'semantic.json.gz',
        data_dir / 'sem_lem.json.gz',
    ]
    datas = {}
    for results_file in results_files:
        feature = results_file.name.split('.')[0]
        cur_data = get_results(data_dir, feature, results_file)
        datas[feature] = cur_data
    return datas


def get_results(data_dir, feature, results_file):
    if results_file.exists():
        with gzip.GzipFile(str(results_file)) as ifh:
            return json.loads(ifh.read())
    if feature == 'sem_lem':
        feature = 'semantic + lemmata'
    payload = search.get_search_params(data_dir, feature)
    return retrieve_results(payload, feature, results_file)


def retrieve_results(payload, feature, results_file):
    source = retrieve_source_name(payload)
    target = retrieve_target_name(payload)
    return search.run_search(payload, source, target, feature, results_file)


def retrieve_source_name(payload):
    return _retrieve_name_core(payload, 'source')


def retrieve_target_name(payload):
    return _retrieve_name_core(payload, 'target')


def _retrieve_name_core(payload, option):
    object_id = payload[option]['object_id']
    r = requests.get(
        f'https://tesserae.caset.buffalo.edu/api/texts/{object_id}/')
    response = r.json()
    assert ('author' in response)
    assert ('title' in response)
    author = response['author']
    title = response['title']
    return f'({author}: {title})'


if __name__ == '__main__':
    _main()
