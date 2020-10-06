from itertools import combinations
from sacrebleu import sentence_bleu
from src.utils import *
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
from src.similarity_models import BertSimilarityModel, SBERTSimilarityModel

OPTS = Config()
for (k, _, v, _) in [
    ("input", str, "data/causalbank/preproc/b32_t32_zyli/train.effect.both",
     "input (training) data file: prefix + sequence train file"),
    ("prefix_file", str, "", 'file containing prefixes to compute (one per line), ranked'),
    ("batch_size", int, 4000, "Batch size"),
    ("gpuid", int, 0, "GPU ID (-1 for CPU)"),
    ("bert_type", str, "sbert", "type of BERT from which to extract embeddings"),
    ("sbert_type", str, 'roberta-large', 'if bert_type is sbert, the base type'),
    ("max_seq_length", int, 128, "model max seq len"),
]: setattr(OPTS, k, v)

## set up device
Device.set_device(OPTS.gpuid)
Device.set_seed(1234)

## set up tokenizer and BERT
print("loading model...")
if OPTS.bert_type in ['bert', 'pbert']:
    sim_model = BertSimilarityModel(OPTS, Device)
elif OPTS.bert_type in ['sbert']:
    sim_model = SBERTSimilarityModel(OPTS, Device)
else:
    raise NotImplementedError()


def div_threshold(resps, div_fn, thresholds):
    """
    :param resps: list of responses
    :param div_fn: function mapping list to  matrix of distances
    :param thresholds: iterable of thresholds to count distinct candidates with
    :return: duplicates: dict mapping thr to lists containing
                         Nones for uniques or index of duplicate
                 counts: dict mapping thr to distinct count

    """
    dists = div_fn(resps)
    duplicates = {}
    counts = {}
    for thr in thresholds:
        ret_t = []
        for i, resp in enumerate(resps):
            res = None
            for j, prev_resp in enumerate(resps[:i]):
                if ret_t[j] is None and dists[i, j] < thr:
                    res = j, round(dists[i, j], 2)
                    break
            ret_t.append(res)
        duplicates[thr] = ret_t
        counts[thr] = sum([ 1 for i in ret_t if i is None])
    return duplicates, counts


def diversity_sbert(output_set, reduce=True):
    set_embs = sim_model.get_embeddings(output_set)
    distances = squareform(pdist(set_embs, 'cosine'))
    itemwise_distances = np.sum(distances, axis=1) / (distances.shape[0] - 1)

    if reduce:
        return np.mean(itemwise_distances)
    else:
        return distances


def diversity_bleu(output_set, n=2, reduce=True):
    if reduce:
        total = 0
        for i in range(len(output_set)):
            total_i = 0
            sent = output_set[i]
            for other_sent in output_set[:i] + output_set[i:]:
                bl = sentence_bleu(sent, other_sent)
                score = bl.bp * math.exp(
                    sum(map(lambda x: math.log(x) if x != 0.0 else -9999999999,
                            bl.precisions[:n])
                        ) / n)
                total_i += 100 - score

            total += total_i

        total = total / (len(output_set) * (len(output_set) - 1))
        return total
    else:
        flatten = lambda l: [item for sublist in l for item in sublist]

        def all_pairs(lst):
            return flatten([[(x, y) for y in lst if x != y] for x in lst])

        def _pair_distance(sent, other_sent):
            bl = sentence_bleu(sent, other_sent)
            return 100 - bl.bp * math.exp(
                sum(map(lambda x: math.log(x) if x != 0.0 else -9999999999,
                        bl.precisions[:n])
                    ) / n)

        distances = np.zeros([len(output_set), len(output_set)])
        for pair in all_pairs(range(len(output_set))):
            distances[pair[0], pair[1]] = _pair_distance(output_set[pair[0]], output_set[pair[1]])
        return distances





examples = [
    "he was expecting it",
    "it was so simple",
    "he was hungry",
    "he knew it was coming",
    "he liked it",
    "he was starving",
    "he was hungry",
    "it was full",
    "it was too crowded",
    "it was dangerous"
]

baseline = [
    "they didn't know what they were doing",
    "they didn't want to lose",
    "they didn't know any better",
    "they didn't know what to expect",
    "they didn't have a chance to win",
    "they didn't have to",
    "they didn't know what to do with it",
    "they knew they were going to win",
    "they didn't want to miss the game",
    "they didn't want to miss a game",
]

from scipy.stats import wilcoxon


def wilcox_test(x, y):
    #     print(x.name,y.name)
    diff_vec = [l - r for (l, r) in zip(x, y) if (float('nan') not in [l, r])]
    if all([d == 0 for d in diff_vec]): return 1
    return (wilcoxon(diff_vec)[-1] if x.name != y.name else 1)


def get_significance_matrix(df):
    """
    returns matrix in which item[y,x] is test of whether y-x is center around zero
    """
    return df.apply(lambda x: (df.apply(lambda y: wilcox_test(x, y))))


if __name__ == "__main__":
    assert diversity_bleu(baseline) - 73.072831 < .001
    assert diversity_bleu(examples) - 83.85799 < .001
    assert diversity_sbert(examples) - 0.58173 < .01
    assert diversity_sbert(baseline) - 0.490297 < .01
