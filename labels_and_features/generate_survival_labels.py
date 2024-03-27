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
import femr.labelers.omop
from femr.extension import datasets as extension_datasets

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

class SurvivalLabeler(Labeler):
    def __init__(self, ontology, codes, required_days):
        self.required_days = required_days
        self.codes = femr.labelers.omop.map_omop_concept_codes_to_femr_codes(ontology, codes)
        print("Converted", len(codes), "to", len(self.codes))
        
    def label(self, patient):
        birth_date = datetime.datetime.combine(patient.events[0].start.date(), datetime.time.min)

        final = -1
        while (patient.events[final].start - patient.events[0].start) > datetime.timedelta(days=365 * 125):
            final -= 1

        censor_time = patient.events[final].start
        
        possible_times = []
        first_history = None
        first_code = None
        
        for event in patient.events:
            if (event.start - birth_date) > datetime.timedelta(days=365 * 125):
                continue

            if event.value is not None or not event.code.startswith('SNOMED/'):
                continue
                
            if first_history is None and (event.start - birth_date) > datetime.timedelta(days=10):
                first_history = event.start
            
            if event.code in self.codes:
                is_event = event.code in self.codes
                if first_code is None and is_event:
                    first_code = event.start
                
            
            if first_history is not None and first_code is None and (event.start - first_history) > datetime.timedelta(days=self.required_days):
                possible_times.append(event.start)
        
        possible_times = [a for a in possible_times if a != first_code]
        if len(possible_times) == 0:
            return []
        
        selected_time = random.choice(possible_times)
        is_censored = first_code is None
        
        if is_censored:
            event_time = censor_time
        else:
            event_time = first_code
        
        survival_value = SurvivalValue(time_to_event=event_time - selected_time, is_censored=is_censored)
        result = [Label(time=selected_time, value=survival_value)]
        return result
    
    def get_labeler_type(self):
        return "survival"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('PATH_TO_FEMR_DB')
parser.add_argument('PATH_TO_SAVE_MATRIX')
parser.add_argument('--NUM_PATIENTS', default=None, type=int)

args = parser.parse_args()

LABELED_PATIENTS = "labels"

NUM_THREADS = 15

with open('../label_defs.pkl', 'rb') as f:
    labels = pickle.load(f)

if __name__ == '__main__':

    # Patient database
    data = femr.datasets.PatientDatabase(args.PATH_TO_FEMR_DB)
    print("Total patients", len(data))

    # Ontology 
    ontology = data.get_ontology()
    
    for name, codes in labels.items():
        if os.path.exists(os.path.join(args.PATH_TO_SAVE_MATRIX, LABELED_PATIENTS, name + ".pickle")):
            continue
        labeler = SurvivalLabeler(ontology, codes, 365)

        labeled_patients = labeler.apply(path_to_patient_database=args.PATH_TO_FEMR_DB, num_threads=NUM_THREADS, num_patients=args.NUM_PATIENTS)
        print("Got", len(labeled_patients), "for", name)
        save_to_file(labeled_patients, os.path.join(args.PATH_TO_SAVE_MATRIX, LABELED_PATIENTS, name + ".pickle"))

        print("Finished Labeling Patients: ", datetime.datetime.now() - start_time)
