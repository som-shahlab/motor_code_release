import pickle

from femr.extension import datasets as extension_datasets
import femr.datasets

from typing import Set
from collections import deque

import argparse
import msgpack

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str)
parser.add_argument('surv_dictionary_path', type=str)
parser.add_argument('target_surv_dictionary_path', type=str)

args = parser.parse_args()

database = femr.datasets.PatientDatabase(args.data)
ontology = database.get_ontology()

codes = set()

with open('label_defs.pkl', 'rb') as f:
    labels = pickle.load(f)
    print(labels)
    for v in labels.values():
        codes |= set(v)


print(codes)

with open(args.surv_dictionary_path, 'rb') as f:
    surv_dict = msgpack.load(f)

last_good = None
for i, code_index in enumerate(surv_dict['codes']):
    code = ontology.get_codes()[code_index]
    if all(c not in codes for c in ontology.get_all_parents(code)):
        last_good = i

print(len(surv_dict['codes']))

num_remapped = 0

# Must have at least one good index
assert last_good is not None

# Reassign tasks to the last good index
for i, code_index in enumerate(surv_dict['codes']):
    code = ontology.get_codes()[code_index]
    if all(c not in codes for c in ontology.get_all_parents(code)):
        last_good = i
    else:
        print(code, [a for a in ontology.get_all_parents(code) if a in codes])
        num_remapped += 1
        surv_dict['codes'][i] = surv_dict['codes'][last_good]
        surv_dict['lambdas'][i] = surv_dict['lambdas'][last_good]

print(num_remapped)

with open(args.target_surv_dictionary_path, 'wb') as f:
    msgpack.dump(surv_dict, f)
