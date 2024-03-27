import pandas
import femr.metrics
import json
import os
import numpy as np
import sklearn
import collections
import random
import pickle
import datetime
import msgpack
import lifelines


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('PATH_TO_SAVE_MATRIX')
parser.add_argument('TARGET_DIR')
parser.add_argument('MOTOR_BATCHES')

args = parser.parse_args()


PATH_TO_SAVE_MATRIX = args.PATH_TO_SAVE_MATRIX
TARGET_DIR = args.TARGET_DIR
MOTOR_BATCHES = args.MOTOR_BATCHES

LABELED_PATIENTS = "labels"
SUBSET_LABELED_PATIENTS = "subset_labels"
FEATURES = "features"

NUM_BOOT = 1000 
#NUM_BOOT = 2

LIMIT = 0.9

print(f'Working with LIMIT {LIMIT}')

with open(args.MOTOR_BATCHES, 'rb') as f:
    config = msgpack.load(f)
    
survival_time_bins = np.array(config['config']['task']['survival_dict']['time_bins']) / (24 * 60)

print(survival_time_bins)

best_results = collections.defaultdict(dict)

targets = []

for target_dir in (TARGET_DIR, 'deep_hit_mimic', 'deep_surv_mimic', 'dsm_mimic',):
    targets += [os.path.join(target_dir, t) for t in os.listdir(target_dir)]


features_path = os.path.join(PATH_TO_SAVE_MATRIX, 'materialized_matrices')

printed = set()

for target in targets:
    if not os.path.exists(os.path.join(target, 'done')):
        continue
    
    with open(os.path.join(target, 'config.json')) as f:
        config = json.load(f)

    data_path = config['path']

    data_path = os.path.join(features_path, os.path.basename(data_path))

    with open(os.path.join(data_path, 'features_config.json')) as f:
        feature_config = json.load(f)

    train_indices = np.load(os.path.join(data_path, 'train_indices.npy'))
    val_indices = np.load(os.path.join(data_path, 'val_indices.npy'))
    test_indices = np.load(os.path.join(data_path, 'test_indices.npy'))
    deltas = np.load(os.path.join(data_path, 'deltas.npy'))
    is_event = np.load(os.path.join(data_path, 'is_event.npy'))

    if feature_config['path'] not in printed:
        printed.add(feature_config['path'])
        print(feature_config['path'], np.sum(is_event), len(is_event) - np.sum(is_event))

    time_limit = np.quantile(deltas[is_event], LIMIT)
    is_event[deltas > time_limit] = 0
    deltas[deltas > time_limit] = time_limit

    def get_nonzero(indices):
        valid = deltas[indices] != 0
        return indices[valid]

    train_indices = get_nonzero(train_indices)
    val_indices = get_nonzero(val_indices)
    test_indices = get_nonzero(test_indices)
    
    # print(feature_config, time_limit, np.mean(deltas[is_event]), np.mean(deltas[test_indices]), np.mean(is_event[test_indices]), len(test_indices))

    if config['models'] == 'survival':
        times = np.load(os.path.join(target, "times.npy"))
        survival = np.load(os.path.join(target, "survival.npy"))
        survival = np.concatenate((survival, np.zeros((survival.shape[0], 1))), axis=-1)
        start = survival[:, :-1]
        end = survival[:, 1:]
        total_times = np.concatenate((times, [float('inf')]))
        time_in_bin = total_times[1:] - total_times[:-1]
        hazard = np.log2(-np.log(end / start) / time_in_bin)
        hazard[:, -1] = hazard[:, -2]

    elif config['models'] == 'cox':
        hazards = np.load(os.path.join(target, 'hazards.npy'))
        hazard = hazards.reshape(-1 ,1)
        hazard = hazard / np.log(2)
        times = [0]
    elif config['models'] == 'deep_surv':
        hazard = np.load(os.path.join(target, 'predictions.npy'))
        hazard = hazard / np.log(2)
        times = [0]

    elif config['models'] == 'deep_hit':
        survival = np.load(os.path.join(target, 'predictions.npy'))
        times = np.load(os.path.join(target, 'bins.npy'))

        survival = np.maximum(survival, -1)

        start = survival[:, :-1]
        end = survival[:, 1:]

        time_in_bin = times[1:] - times[:-1]

        delta = -np.log(end / start)
        hazard = np.log2(delta / time_in_bin)
        times = times[:-1]
    elif config['models'] == 'dsm':
        times = np.load(os.path.join(target, "bins.npy"))
        hazard = np.load(os.path.join(target, "hazard.npy"))

        times = times[:-1]

    times[0] = 0
    def get_metrics(indices):
        main_statistic = femr.extension.metrics.compute_c_statistic(deltas[indices], ~is_event[indices], times, hazard[indices, :])[0]
        return main_statistic

    key = config['models']
    sub_dict = best_results[feature_config['path']]

    results = {
        'train': get_metrics(train_indices),
        'val': get_metrics(val_indices),
        'train_indices': train_indices,
        'test_indices': test_indices,
        'deltas': deltas,
        'is_event': is_event,
        'times': times,
        'hazard': hazard,
        'config': config,

    }

    if key not in sub_dict or results['val'] > sub_dict[key]['val']:
        sub_dict[key] = results


table_entry = collections.defaultdict(dict)

def standard_cdf(deltas, times, hazard, eval_times):
    total_times = np.concatenate((times, [float('inf')])).reshape(1, -1)
    time_in_bin = np.clip(eval_times.reshape(-1, 1) - total_times[:, :-1], 0, total_times[:, 1:] - total_times[:, :-1])

    total_hazard = np.exp2(hazard) @ time_in_bin.T

    probs = 1 - np.exp(-total_hazard)
    return probs

def breslow_cdf(deltas, is_event, times, hazard, train_indices, eval_time):
    breslow = femr.metrics.estimate_breslow(deltas[train_indices], ~is_event[train_indices], times, hazard[train_indices, :])
    cdf = np.array(femr.metrics.apply_breslow(np.ones_like(deltas) * eval_time, times, hazard, breslow))
    return cdf

better = True
print('BETTER ', better)

def get_probs(m, times, hazard, deltas, is_event, train_indices):
    t = np.median(deltas[is_event])
    if m in ['cox', 'deep_surv']:
        return breslow_cdf(deltas, is_event, times, np.exp2(hazard), train_indices, t)
    else:
        return standard_cdf(deltas, times, hazard, np.array([t])).reshape(-1)

def get_times_hazards(k, m):
    if m in ['cox', 'survival', 'deep_surv', 'deep_hit', 'dsm']:
        if m not in best_results[k]:
            return None
        v = best_results[k][m]
        times = v['times']
        hazard = v['hazard']
        return times, hazard
    else:
        with open(os.path.join(m + '_predictions', k + '_saved.pkl'), 'rb') as f:
            predictions = pickle.load(f)
            
        features_path = os.path.join(PATH_TO_SAVE_MATRIX, FEATURES, k + '.pickle')

        with open(features_path, 'rb') as f:
            data = pickle.load(f)


        labels_path = os.path.join(PATH_TO_SAVE_MATRIX, SUBSET_LABELED_PATIENTS, k + ".pickle")

        with open(labels_path, 'rb') as f:
            label_obj = pickle.load(f)


        assert len(data['labels']) == len(label_obj)

        pids = data['pids']
        label_times = data['times']

        used_indices = set()

        hazard = []
        empty = np.zeros(len(survival_time_bins))

        for i in range(len(pids)):
            pid = pids[i]
            time = label_times[i].astype(datetime.datetime)
            used_indices.add((pid, time))
            if (pid, time) not in predictions:
                hazard.append(empty)
            else:
                _, h, delta, event = predictions[(pid, time)]
                hazard.append(h)

        hazard = np.stack(hazard)

        for a in predictions:
            if a not in used_indices:
                print("missing?", k, a)
                print(a[0] in pids, label_obj[a[0]])

        times = survival_time_bins
        
        hazard += np.log2(60 * 24)

        return times, hazard

with open('foo', 'w') as f:
    for k, vs in best_results.items():
        for m, r in vs.items():
            print(k, m, r['config'], r['train'], r['val'])
            if m == 'survival':
                f.write(json.dumps(r['config']) + '\n')

tasks = ('celiac_disease', 'heart_attack', 'lupus', 'NAFL', 'pancreatic_cancer', 'stroke')

def generate_table(metric, fancy_names):
    baseline_method = list(fancy_names.keys())[-1]
    baselines = {}
    results = {}

    print("Creating table for ", metric)
    for m in fancy_names:
        for k in tasks:
            if k not in best_results:
                results[(m, k)] = None
                continue
            r = best_results[k]

            baseline = 'survival'
            v = r[baseline]

            train_indices = v['train_indices']
            test_indices = v['test_indices']
            deltas = v['deltas']
            is_event = v['is_event']

            if metric == 'rank':
                def compute(indices, times, hazard, probs):
                    return femr.metrics.compute_c_statistic(deltas[indices], ~is_event[indices], times, hazard[indices, :])
            elif metric == 'harrell':
                def compute(indices, times, hazard, probs):
                    average_hazard = np.mean(hazard, axis=1)
                    return lifelines.utils.concordance_index(deltas[indices], -average_hazard[indices], is_event[indices]), None
            elif metric == 'calibrate-1':
                def compute(indices, times, hazard, probs):
                    d = deltas[indices]
                    p = probs[indices]
                    e = is_event[indices]

                    sort_indices = np.argsort(p)

                    statistic = 0

                    t = np.median(deltas[is_event])

                    for chunk in np.array_split(sort_indices, 10):
                        kmf = lifelines.KaplanMeierFitter()
                        kmf.fit(d[chunk], e[chunk])

                        expected = float(kmf.cumulative_density_at_times(t))
                        actual = np.mean(p[chunk])

                        value = (expected - actual) ** 2 / (actual * ( 1 - actual))
                      
                        statistic += value

                    return statistic, None

            if k not in baselines:
                indices_to_eval = []
                baseline_perf = []

                times, hazard = get_times_hazards(k, baseline_method)
                probs = get_probs(baseline_method, times, hazard, deltas, is_event, test_indices)

                rng = random.Random(125312)

                for i in range(NUM_BOOT):
                    sample = sklearn.utils.resample(test_indices, random_state = rng.getrandbits(32))
                    indices_to_eval.append(sample)
                    sub_statistic = compute(sample, times, hazard, probs)[0]
                    baseline_perf.append(sub_statistic)
                baselines[k] = (indices_to_eval, baseline_perf)

            indices_to_eval, baseline_perf = baselines[k]

            times_hazards = get_times_hazards(k, m)

            if times_hazards is None:
                results[(m, k)] = None
            else:
                times, hazard = times_hazards
                probs = get_probs(m, times, hazard, deltas, is_event, train_indices)

                main_statistic, values = compute(test_indices, times, hazard, probs)
                
                samples = []
                for i in range(len(indices_to_eval)):
                    sample = indices_to_eval[i]
                    sub_statistic = compute(sample, times, hazard, probs)[0]
                    samples.append(sub_statistic - baseline_perf[i])

                pct =  np.quantile(samples, [0.025, 0.975])
                results[(m, k)] = (main_statistic, pct)
                # print(pct)
                # print(k, m, len(indices), main_statistic, pct)
        
    def comp_str(a):
        if metric == 'calibrate-1':
            a = -a
        return a

    main_table = []
    relative_table = []

    final_best_results = {}
    for k in tasks:
        for m in fancy_names:
            val = comp_str(results[(m, k)][0])

            if k not in final_best_results or val > final_best_results[k]:
                final_best_results[k] = val

    for m, name in fancy_names.items():
        text_entries = [name]

        for k in tasks:
            val = results[(m, k)][0]
            str_val = f'{val:.3f}'
            if final_best_results[k] == comp_str(val):
                str_val = '\\textbf{' + str_val + '}'
            text_entries.append(str_val)
        
        main_table.append(' & '.join(text_entries) + r'\\')

        text_entries = [name]

        for k in tasks:
            start, end = results[(m, k)][1]
            str_val = f'[{start:.3f}, {end:.3f}]'
            if end < 0 or start > 0:
                str_val =  str_val + '\\text{*} '
            text_entries.append(str_val)

        relative_table.append(' & '.join(text_entries) + r'\\')

    print("Main table")
    print('\n'.join(main_table))
    print("Relative table")
    print('\n'.join(relative_table))

 
fancy_name = {
    'cox': 'Cox PH',
    'deep_surv': 'DeepSurv',
    'dsm': 'DSM',
    'deep_hit': 'DeepHit',
    'survival': 'RSF',
    #'scratch': 'MOTOR-Scratch',
    'motor_probe': 'MOTOR-Probe',
    #'motor_finetune': 'MOTOR-Finetune',
}
generate_table('rank', fancy_name)
#generate_table('calibrate-1', fancy_name)
#generate_table('harrell', fancy_name)

print(1/0)

fancy_name = {
    'next_code_finetune': 'Next Code Pretraining',
    'motor_finetune': 'Time-to-Event Pretaining',
}
generate_table('rank', fancy_name)

fancy_name = {
    'motor_probe_subsample_5': '5\\%',
    'motor_probe_subsample_10': '10\\%',
    'motor_probe_subsample_25': '25\\%',
    'motor_probe': '100\\%',
}

generate_table('rank', fancy_name)

fancy_name = {
    'motor_5_probe': '5\\%',
    'motor_10_probe': '10\\%',
    'motor_25_probe': '25\\%',
    'motor_probe': '100\\%',
}

generate_table('rank', fancy_name)
