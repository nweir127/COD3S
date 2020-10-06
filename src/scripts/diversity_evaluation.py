from argparse import ArgumentParser
import numpy as np
import math
import pprint
from itertools import product
import re
import os
import pandas as pd
from tabulate import tabulate

DF = pd.DataFrame

parser = ArgumentParser()

data_args = parser.add_argument_group('Dataset')
data_args.add_argument('--tasks', type=str, nargs='+',
                       choices=['c2e', 'e2c'], default=['c2e', 'e2c'])
data_args.add_argument('--dataset', type=str, nargs='+',
                       choices=['copa', 'causalbank'], default=['copa','causalbank'])
data_args.add_argument('--results_dir', type=str, default='exp')
# data_args.add_argument('--human_eval_data', type=str, default="exp/human_eval_data/Baseline.baseline")
data_args.add_argument('--out_dir', type=str, default='auto_eval')


stat_args = parser.add_argument_group('Statistical Test Parameters')
stat_args.add_argument('--sig_p', type=float, default=.05)

args = parser.parse_args()
import pickle
from src.utils.metrics import *
from pathlib import Path
Path(args.out_dir).mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(filename=f'{args.out_dir}/results.log',level=logging.DEBUG)
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f'{args.out_dir}/results.txt', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

for (task, dataset) in product(args.tasks, args.dataset):
    print("TASK: {}\tDATASET:{}".format(task, dataset))
    if task == 'c2e':
        source, target = ('cause', 'effect')
    else:
        source, target = ('effect', 'cause')

    def respath(expdir, task, dataset, hamming, pinf, rs, sinf, srep=None, trep=None):
        if expdir == 'baseline_generate':
            ret=f"baseline_generate/Direction.{task}+DownloadPreprocessed.yes+EvalData.{dataset}"
            ret+=f"+RandomSampling.{rs}+SourceRep.{srep}+TargetRep.{trep}/results"
        else:
            ret = f"generate_mmi/Bits.16+Debug.no+Direction.{task}"
            ret += f"+DownloadPreprocessed.yes+EvalData.{dataset}"
            ret += f"+HammingThreshold.{hamming}+PrefInference.{pinf}"
            ret += f"+RandomSampling.{rs}+SeqInference.{sinf}+SourceRep.seq+TargetRep.both/out"

        return ret


    #### load cod3s results
    file_pairs = [
        (respath('baseline_generate', task, dataset, None, None, 'no', None, srep='seq',trep='seq'), 'seq2seq-baseline'),
        (respath('baseline_generate', task, dataset, None, None, 'no', None, srep='seq',trep='both'), 'seq2both-baseline'),
        (respath('baseline_generate', task, dataset, None, None, 'yes', None, srep='seq',trep='seq'), 's2s-rs'),
        (respath('generate_mmi', task, dataset, 'yes', 'beam', 'no', 'beam'), 'beam-beam'),
        # (respath('generate_mmi', task, dataset, 'yes', 'bidi', 'no', 'beam'), 'bidi-beam'),
        (respath('generate_mmi', task, dataset, 'yes', 'beam', 'no', 'bidi'), 'beam-bidi'),
        (respath('generate_mmi', task, dataset, 'no', 'beam', 'no', 'bidi'), 'beam-bidi-no_div'),
        (respath('generate_mmi', task, dataset, 'yes', 'beam', 'yes', 'bidi'), 'beam-bidi-rs'),
        (respath('generate_mmi', task, dataset, 'yes', 'bidi', 'no', 'bidi'), 'bidi-bidi'),
        (respath('generate_mmi', task, dataset, 'no', 'bidi', 'no', 'bidi'), 'bidi-bidi-no_div'),
        (respath('generate_mmi', task, dataset, 'yes', 'bidi', 'yes', 'bidi'), 'bidi-bidi-rs'),
        (respath('generate_mmi', task, dataset, 'no', 'bidi', 'yes', 'bidi'), 'bidi-bidi-rs-no_div'),
    ]

    print("Evaluated File Pairs:")
    for fp in file_pairs:
        print("{}\t{}".format(fp[1],fp[0]))


    def read_predictions(results_file, label):
        items = []
        line_idx = 0
        for line in read_sentences(results_file, progbar=False):
            if re.match('^[H|S|O|T]-', line):
                linetype = line[0]
                if linetype == 'S':
                    line_idx = 0
                    inp = line.split('\t')[-1].lower()
                res = {
                    'id': int(re.search(f'{line[0]}-(\d+)', line).group(1)),
                    'type': linetype,
                    'label': label,
                    "line_idx": line_idx,
                    'input': inp
                }
                line_idx += 1
                if "$SEP$" in line:
                    res['prefix'] = re.findall(r"(\$.+\$\s)+", line)[0]
                    res['sequence'] = line.split('$')[-1].strip()
                else:
                    res['sequence'] = line.split('\t')[-1]

                scores = list(map(float, re.findall('\t(-[0-9\.]+)', line)))[::-1]
                for k, sc in zip(['sequence_score', 'prefix_score'], scores):
                    res[k] = sc

                items.append(res)
        return DF(items)


    cod3_results = pd.concat([
        read_predictions(f"{args.results_dir}/{f}/generate_cands.log", label) for (f, label) in file_pairs
    ])
    cod3_results = cod3_results[cod3_results.type == 'H']

    #### load zyli baselines
    data_dir = "{}/human_eval_data/Baseline.baseline/{}/{}".format(args.results_dir,f"{source}2{target}", (dataset if dataset=='copa' else 'cb'))
    all_responses = []
    for my_model in ['cons', 'rand']:
        for file in os.listdir(data_dir):
            if file.startswith(my_model):
                label = file
                all_responses += [
                    (i, my_model, label, inp.lower(), line)
                    for i, (line, inp) in
                    enumerate(zip(
                        read_sentences("{}/{}".format(data_dir, file), progbar=False),
                        read_sentences("{}/{}".format(data_dir, "input"), progbar=False)
                    )) if line
                ]

    zy_responses = DF([x for x in all_responses if 'ATTN' not in x[3]],
                      columns=['id', 'label', 'rank', 'input', 'sequence'])

    all_results = pd.concat([cod3_results, zy_responses])

    diversity_scores = DF()
    tenset_diversity_scores = DF()
    for (metric, metric_fn) in [
        ("1-BLEU", lambda s: diversity_bleu(s, n=1)),
        ("2-BLEU", lambda s: diversity_bleu(s, n=2)),
        ("SBERT", lambda s: diversity_sbert(s)),
    ]:
        logging.info("Getting Triplet {} Diversity Scores...".format(metric))
        diversity_scores[metric] = all_results.groupby(['label', 'id']).apply(
            lambda x: metric_fn(list(x.sequence)[:3]) if len(x) > 1 else float('nan'))

        logging.info("Getting 10-set {} Diversity Scores...".format(metric))
        tenset_diversity_scores[metric] = cod3_results.groupby(['label', 'id']).apply(
            lambda x: metric_fn(list(x.sequence)[:10]))
    # sort_values(by=('prefix_score' if 'beam' in label or 'bidi' in label else 'sequence_score'), ascending=False)


    metric_fn = lambda s: diversity_sbert(s, reduce=False)
    METRICS = ['1-BLEU', '2-BLEU', 'SBERT']
    # thresholds = np.arange(0, .75, .05)
    thresholds = [0.00, 0.1, 0.25, 0.50, 0.75]


    threshold_3 = DF([
        {
            **{'id': id, "label": label},
            **(div_threshold(list(x.sequence)[:3], metric_fn, thresholds = thresholds)[1])
        }
        for (label, id), x in all_results.groupby(['label', 'id'])
        if len(x) >= 3
    ])
    threshold_10 = DF([
        {
            **{'id': id, "label": label},
            **(div_threshold(list(x.sequence)[:10], metric_fn, thresholds = thresholds)[1])
        }
        for (label, id), x in cod3_results.groupby(['label', 'id'])
        if len(x) >= 10
    ])

    for div_mat, thr_mat, label in [
        (diversity_scores, threshold_3, "Triplets"),
        (tenset_diversity_scores, threshold_10, "TenSets")
    ]:
        print(f"\n===================== {dataset} {task} {label} DIVERSITY =====================\n")
        grouped = div_mat.groupby(level=0)
        means = grouped.mean()

        sig_mats = {}
        p = args.sig_p


        for m in METRICS:
            sig_mats[m] = get_significance_matrix(div_mat.unstack(level=0)[m])

            sig_mats[m].to_pickle(f"{args.out_dir}/{dataset}_{task}_{label}_{m}_div_significance.pkl")


        print(means.apply(lambda row: "{:.1f} / {:.1f} / {:.3f}".format(
            row['1-BLEU'], row['2-BLEU'], row['SBERT']), axis=1))

        means.to_pickle(f"{args.out_dir}/{dataset}_{task}_{label}_div_metric_means.pkl")



        for m in METRICS:
            print('')
            print(f"================== {dataset} {task} {label} {m} SIGNIFICANCE ================== ")
            print('')
            print(tabulate(sig_mats[m], floatfmt=".2f", headers='keys'))
            print('')

        thr_grouped = thr_mat.groupby(['label','id']).mean()
        thr_means = thr_grouped.groupby(level=0).mean()
        thr_means.columns = ['{:.2f}'.format(c) if type(c) != str else c for c in thr_means.columns]
        print('')
        print(f"=============== {dataset} {task} {label} SBERT discrete counts =============== ")
        print('')
        print(tabulate(thr_means, headers='keys'))
        thr_means.to_pickle(f"{args.out_dir}/{dataset}_{task}_{label}_discrete_counts.pkl")

        # print(f"================== {dataset} {task} {label} {m} SBERT_DISCRETE ================== ")
        # print('')
        # print(tabulate(sig_mats['SBERT_DISCRETE'], floatfmt=".2f", headers='keys'))
        # print('')

    print('\n\n')