import subprocess
import hashlib
import os
import json
import sys
import pickle
import numpy as np
import multiprocessing
import datetime

nodesizes = [5, 15, 50, 100]
num_bins = [8, 16, 32]

NUM_THREADS = 1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('PATH_TO_SAVE_MATRIX')
parser.add_argument('outdir')
parser.add_argument('index', type=int)

args = parser.parse_args()

index = args.index

PATH_TO_SAVE_MATRIX = args.PATH_TO_SAVE_MATRIX
LABELED_PATIENTS = "labels"
SUBSET_LABELED_PATIENTS = "more_subset_labels"
FEATURES = "features"

features_path = os.path.join(PATH_TO_SAVE_MATRIX, 'materialized_matrices')
outdir = args.outdir

if not os.path.exists(outdir):
    os.mkdir(outdir)

def run(config):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    outpath = os.path.join(outdir, config_hash)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    with open(os.path.join(outpath, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if os.path.exists(os.path.join(outpath, 'done')):
        return

    start = datetime.datetime.now()

    with open(os.path.join(outdir, 'start'), 'w') as f:
        f.write(start.isoformat() + '\n')

    config['outpath'] = outpath

    args = ['Rscript', 'baselines/train_survival_model.R']
    for a, b in config.items():
        args.append('--' + a)
        args.append(str(b))
        
    e = dict(os.environ)
        
    e['RF_CORES'] = '16'
    e['OMP_NUM_THREADS'] = '16'
   
    subprocess.run(args, env=e)

    end = datetime.datetime.now()

    with open(os.path.join(outdir, 'end'), 'w') as f:
        f.write(end.isoformat() + '\n')

desired = []
features = os.listdir(features_path)
for path in [os.path.join(features_path, path) for path in features]:
    desired.append({'path': path, 'models': 'cox'})

    for num_bin in num_bins:
        for nodesize in nodesizes:
            desired.append({'path': path, 'models': 'survival', 'num_bin': num_bin, 'nodesize': nodesize})


import random
random.seed(2312)
random.shuffle(desired)
print(len(desired))

if index < len(desired):
    target = desired[index]

    print(target)
    run(target)
