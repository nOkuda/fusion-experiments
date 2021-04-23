import gzip
import json
import time

import requests

API_URL = 'https://tesserae.caset.buffalo.edu/api/'


def get_search_params(data_dir, feature):
    outname = {
        'lemmata': 'lemmata',
        'semantic': 'semantic',
        'semantic + lemmata': 'sem_lem',
    }
    filepath = data_dir / f'params.{outname[feature]}.json'
    if filepath.exists():
        with open(str(filepath)) as ifh:
            return json.load(ifh)
    features2stopwords = {
        'lemmata': [
            "qui", "quis", "et", "sum", "in", "is", "non", "hic", "ego", "ut",
            "ad", "ille", "quod", "dico", "cum"
        ],
        'semantic': [
            "qui", "quis", "et", "sum", "in", "is", "non", "hic", "ego", "ut",
            "ad", "ille", "quod", "dico", "cum"
        ],
        'semantic + lemmata': [
            "et", "qui", "facio", "quis", "tat", "sum", "is", "non", "in",
            "propterea", "nonne", "ni", "hic", "atqui", "sicut", "hornotinus",
            "sitanius", "tamquam", "ab", "ego"
        ],
    }
    source = 'aeneid'
    target = 'lucan'
    source_id = _retrieve_work_id('vergil', 'aeneid')
    target_id = _retrieve_work_id('lucan', 'bellum civile')
    search_type = 'phrase'
    stopwords = features2stopwords[feature]
    print(f'Generating payload for {source} {target}')
    score_basis = feature if search_type == 'selfscore' else 'lemmata'
    payload = {
        'source': {
            'object_id': source_id,
            'units': 'phrase'
        },
        'target': {
            'object_id': target_id,
            'units': 'phrase'
        },
        'method': {
            'name': 'original',
            'feature': feature,
            'stopwords': stopwords,
            'score_basis': score_basis,
            'freq_basis': 'texts',
            'max_distance': 50,
            'distance_basis': 'frequency',
            'min_score': 0
        }
    }
    par_dir = filepath.parent
    if not par_dir.exists():
        par_dir.mkdir(parents=True, exist_ok=True)
    with open(str(filepath), 'w') as ofh:
        json.dump(payload, ofh)
    return payload


def run_search(payload, source, target, feature, outfilename):
    r = requests.post(f'{API_URL}parallels/',
                      json=payload,
                      allow_redirects=False)
    if r.status_code == 201:
        assert 'Location' in r.headers
        final_url = r.headers['Location']
    elif r.status_code == 303:
        assert 'Location' in r.headers
        final_url = r.headers['Location']
        final_url = final_url.split('?')[0]
    else:
        print(r.url)
        print(r.status_code)
        print(r.json())
        raise Exception('Bad request')
    ok = _wait_until_done(final_url, source, target, feature)
    if ok:
        print(f'Saving results {source} {target} {feature}')
        options = {
            'sort_by': 'score',
            'sort_order': 'descending',
            'per_page': 50000,
            'page_number': 0
        }
        results = requests.get(final_url, params=options).json()
        while len(results['parallels']) != results['total_count']:
            print(len(results['parallels']), results['total_count'],
                  len(results['parallels']) / results['total_count'])
            options['page_number'] = options['page_number'] + 1
            cur_results = requests.get(final_url, params=options).json()
            results['parallels'].extend(cur_results['parallels'])
        with gzip.open(str(outfilename), 'wt', encoding='utf-8') as ofh:
            json.dump(results, ofh)
        return results
    raise Exception('Could not get results')


def _retrieve_work_id(author, title):
    r = requests.get(f'{API_URL}texts/',
                     params={
                         'author': author,
                         'title': title
                     })
    response = r.json()
    assert ('texts' in response)
    assert (len(response['texts']) == 1)
    return response['texts'][0]['object_id']


def _wait_until_done(final_url, source, target, feature):
    status_url = final_url + 'status/'
    print(f'Waiting for {source} {target} {feature} ({status_url})')
    while True:
        time.sleep(1.0)
        r = requests.get(status_url)
        response = r.json()
        assert 'status' in response
        if response['status'] == 'Done':
            return True
        elif response['status'] == 'Failed':
            print(f'Failed: {source} {target} {feature}')
            print(r.json())
            return False
