import datetime
import os
from typing import List, Tuple, Set

import numpy as np
from sklearn import metrics

import femr
import femr.datasets
from femr.labelers.core import Label, LabeledPatients, TimeHorizon, Labeler, SurvivalValue
from femr.featurizers.core import Featurizer, FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.extension import datasets as extension_datasets

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# import xgboost as xgb
import pickle
import datetime
from collections import deque
import random

start_time = datetime.datetime.now()

def save_to_file(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)

def load_from_file(path_to_file: str):
    """Load object from Pickle file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result

# Please update this path with your extract of femr as noted in previous notebook. 
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

NUM_THREADS = 24

import random

if __name__ == '__main__':
    # Patient database
    data = femr.datasets.PatientDatabase(args.PATH_TO_FEMR_DB)

    # Ontology 
    ontology = data.get_ontology()

    for name in os.listdir(os.path.join(args.PATH_TO_SAVE_MATRIX, LABELED_PATIENTS)):
        name = name.split('.')[0]
        print("Generating for ", name)
        labeled_patients = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, SUBSET_LABELED_PATIENTS, name + ".pickle"))

        if os.path.exists(os.path.join(PATH_TO_SAVE_MATRIX, FEATURES, name + ".pickle")):
            print("Already done", name)
            continue

        age = AgeFeaturizer()
        count = CountFeaturizer()
        featurizer_age_count = FeaturizerList([age, count])
        
        featurizer_age_count.preprocess_featurizers(args.PATH_TO_FEMR_DB, labeled_patients, num_threads=NUM_THREADS)
        full_matrix, pids, labels, times = featurizer_age_count.featurize(args.PATH_TO_FEMR_DB, labeled_patients, num_threads=NUM_THREADS)
        
        features = {'full_matrix': full_matrix, 'labels': labels, 'pids': pids, 'times': times}
        save_to_file(features, os.path.join(args.PATH_TO_SAVE_MATRIX, FEATURES, name + ".pickle"))
