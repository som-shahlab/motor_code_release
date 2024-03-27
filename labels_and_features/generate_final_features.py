import scipy.linalg
import hashlib
import os
import json
import pickle
import numpy as np
import multiprocessing
import femr.datasets
import datetime

min_patients = [10, 100, 1000]

NUM_THREADS = 1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('PATH_TO_FEMR_DB')
parser.add_argument('PATH_TO_SAVE_MATRIX')

args = parser.parse_args()

PATH_TO_FEMR_DB = args.PATH_TO_FEMR_DB
PATH_TO_SAVE_MATRIX = args.PATH_TO_SAVE_MATRIX


LABELED_PATIENTS = "labels"
SUBSET_LABELED_PATIENTS = "more_subset_labels"
FEATURES = "features"

targets = [a.split('.pickle')[0] for a in os.listdir(os.path.join(PATH_TO_SAVE_MATRIX, 'labels'))]

print(targets)

outdir = os.path.join(args.PATH_TO_SAVE_MATRIX, 'materialized_matrices')

if not os.path.exists(outdir):
    os.mkdir(outdir)

vals = []

database = femr.datasets.PatientDatabase(PATH_TO_FEMR_DB)

def run(config):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    outpath = os.path.join(outdir, config_hash)
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    features_path = os.path.join(PATH_TO_SAVE_MATRIX, FEATURES, config['path'] + '.pickle')

    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    features = data['full_matrix']
    labels = data['labels']
    pids = data['pids']
    times = data['times']

    unique, counts = np.unique(features.nonzero()[1], return_counts=True)
    valid = counts >= config['min_patient']

    valid_columns = unique[valid]

    hashed_pids = np.array([database.compute_split(97, pid) for pid in pids])

    train_indices  = (hashed_pids < 70).nonzero()[0]
    val_indices  = ((hashed_pids >= 70) & (hashed_pids < 85)).nonzero()[0]
    test_indices  = ((hashed_pids >= 85) & (hashed_pids < 100)).nonzero()[0]

    print(len(train_indices), len(val_indices), len(test_indices))

    features = features.tocsc()[:, valid_columns].toarray()
    deltas = np.array([label.time_to_event / datetime.timedelta(days=1) for label in labels])
    is_event = np.array([not label.is_censored for label in labels])

    r = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'features': features,
        'deltas': deltas,
        'is_event': is_event,
        'times': times,
        'pids': pids,
    }

    for k, v in r.items():
        with open(os.path.join(outpath, k + '.npy'), 'wb') as f:
            np.save(f, v)

    with open(os.path.join(outpath, 'features_config.json'), 'w') as f:
        json.dump(config, f)
    
    config['outpath'] = outpath

for path in targets:
    for min_patient in min_patients:
        config = {'min_patient': min_patient, 'path': path}
        run(config)
