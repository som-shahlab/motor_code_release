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

LABELED_PATIENTS = "labels"
LESS_SUBSET_LABELED_PATIENTS = "subset_labels"
SUBSET_LABELED_PATIENTS = "more_subset_labels"


import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('PATH_TO_SAVE_MATRIX')
# Please update this path with your extract of femr as noted in previous notebook. 

args = parser.parse_args()
PATH_TO_SAVE_MATRIX = args.PATH_TO_SAVE_MATRIX

if __name__ == '__main__':
    for name in os.listdir(os.path.join(args.PATH_TO_SAVE_MATRIX, LABELED_PATIENTS)):
        name = name.split('.')[0]
        labeled_patients = load_from_file(os.path.join(args.PATH_TO_SAVE_MATRIX, LABELED_PATIENTS, name + ".pickle"))

        censored_labels = []
        event_labels = []
        
        num_events = 0
        for pid, labels in labeled_patients.items():
            if len(labels) == 0:
                continue
            assert len(labels) == 1
            if labels[0].value.is_censored:
                censored_labels.append((pid, labels))
            else:
                event_labels.append((pid, labels))
        
        print(name, len(event_labels), len(event_labels) * 4)
        
        random.shuffle(censored_labels)
        sub_censored_labels = censored_labels[:len(event_labels) * 4]
        
        final_labels = {}
        for pid, labels in (event_labels + sub_censored_labels):
            final_labels[pid] = labels
        
        final = LabeledPatients(final_labels, labeled_patients.get_labeler_type())
        print("Final large size" ,len(final), name)
        save_to_file(final, os.path.join(PATH_TO_SAVE_MATRIX, LESS_SUBSET_LABELED_PATIENTS, name + ".pickle"))

        if len(event_labels) > 40_000:
            event_labels = random.sample(event_labels, 40_000)
        
        random.shuffle(censored_labels)
        sub_censored_labels = censored_labels[:len(event_labels) * 4]
        
        final_labels = {}
        for pid, labels in (event_labels + sub_censored_labels):
            final_labels[pid] = labels
        
        final = LabeledPatients(final_labels, labeled_patients.get_labeler_type())
        print("Final size" ,len(final), name)
        save_to_file(final, os.path.join(PATH_TO_SAVE_MATRIX, SUBSET_LABELED_PATIENTS, name + ".pickle"))
