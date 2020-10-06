global Device
from src.utils import *
from src.utils.read_data import SentenceDataset
from collections import defaultdict
from src.sbert_lsh_model import SBERTLSHModel, BertLSHModel
import os

OPTS = Config.from_command_line(
    arg_defaults=[
        ##### I/O ####
        ("input", str, "", "input data file: source - target tsv"),
        ("split", str, "", "split"),
        ("output_dir", str, "", "output path"),
        #### BERT #####
        ("bert_type", str, "sbert", "type of BERT from which to extract embeddings"),
        ("sbert_type", str, 'roberta-large', 'if bert_type is sbert, the base type'),
        ("batch_size", int, 32, "Batch size"),
        ("gpuid", int, -1, "GPU ID (-1 for CPU)"),
        ("max_seq_length", int, 20, "model max seq len"),
        #### MULTI JOBS ####
        ("local_rank", int, -1, "batch displacement for distributed feature extraction"),
        ("num_threads", int, 1, ""),
        #### SIGNATURE OPTS #####
        ("lsh_dim", int, 10, "number of base vectors for LSH hash (= number bits in signature)"),
        ("sep_token", str, "$SEP$", "separating token"),
        #### SOURCE/TARGET PARAMS ####
        ("source_repr", str, "seq", "representation of source {seq, both, prefix}. for multiple concat w/comma"),
        ("target_repr", str, "both", "representation of target {seq, both, prefix}. for multiple concat w/comma"),
        ("source_lang", str, "cause", "source language"),
        ("target_lang", str, "effect", "source language"),
        #### MISC ####
        ("debug", int, 0, "debug"),
        ("seed", int, 1234, "random seed"),
        ("remove_end_punc", int, 1, 'whether to strip the period from the end (for copa examples)'),
        ('exclude_prefix_patterns', int, 1, 'hacky exclusion of effects with ill-formed patterns from causalbank'),
        ('example_bins', int, 0, 'print bins with most examples'),
    ],
    description="prepends bit signatures to training/dev/test data"
)
for req_attr in [
    'input', 'split', 'output_dir'
]:
    assert not not getattr(OPTS, req_attr)

from pathlib import Path
Path(OPTS.output_dir).mkdir(parents=True, exist_ok=True)

## set up device
Device.set_device(OPTS.gpuid)
Device.set_seed(OPTS.seed)


## set up tokenizer and BERT
print("loading model...")
if OPTS.bert_type in ['bert']:
    sim_model = BertLSHModel(OPTS, Device)
elif OPTS.bert_type in ['sbert']:
    sim_model = SBERTLSHModel(OPTS, Device)
else:
    raise NotImplementedError()

## initialize dataset reader/loader
print('initializing dataset...')
if OPTS.num_threads > 1:
    every = OPTS.num_threads
    assert (OPTS.local_rank > 0 and OPTS.local_rank <= OPTS.num_threads)
    displace = OPTS.local_rank - 1
else:
    every = 1
    displace = 0
dataset = SentenceDataset(OPTS.input, every=every, displace=displace)

loader = tsv_reader(dataset, OPTS)

initialized_lsh = False
signature_counts = defaultdict(lambda: defaultdict(int))

writers = defaultdict(dict)
for side, opt in zip([OPTS.source_lang, OPTS.target_lang], [OPTS.source_repr, OPTS.target_repr]):
    for repr in opt.split(','):
        repr = repr.strip()
        out_file = f"{OPTS.split}.{side}.{repr}"
        writers[side][repr] = open(os.path.join(OPTS.output_dir, out_file), 'w', encoding='utf-8')

if OPTS.example_bins:
    ex_strings = defaultdict(list)
else:
    ex_strings = None

skipped_instances = 0

for i, pair in enumerate(loader):
    (source_batch, target_batch) = pair

    source_batch, target_batch = zip(*[
        (t, s) for (t, s) in zip(source_batch, target_batch)
        if len(t.split()) <= OPTS.max_seq_length and len(s.split()) <= OPTS.max_seq_length
    ])

    bert_features = {}

    if any([repr in OPTS.target_repr for repr in ['both', 'prefix']]):
        bert_features[OPTS.target_lang] = sim_model.get_embeddings([
            well_formed_sentence(t) for t in target_batch
        ])
        if i == 0 and OPTS.debug:
            ppr(target_batch[:10])
    if any([repr in OPTS.source_repr for repr in ['both', 'prefix']]):
        bert_features[OPTS.source_lang] = sim_model.get_embeddings([
            well_formed_sentence(s) for s in source_batch
        ])
        if i == 0 and OPTS.debug:
            ppr(source_batch[:10])

    if not initialized_lsh:
        sim_model.hasher.reset(
            dim=bert_features[OPTS.target_lang if
            OPTS.target_lang in bert_features
            else OPTS.source_lang].shape[-1]
        )
        initialized_lsh = True

    signatures = {
        k: [sim_model.hasher.hash_vector(output)[0] for output in bert_features[k]]
        for k in bert_features
    }

    for b_idx, (source, target) in enumerate(zip(source_batch, target_batch)):

        ### hack to handle ill formed causalbank bug
        if OPTS.exclude_prefix_patterns:
            proc_effect = target.lower().strip()
            if any([proc_effect.endswith(x) for x in [
                    'be',
                    'maybe',
                    'actually',
                    'partly',
                    'it is',
                    'it was',
                    'that was',
                    'that is',]]):
                skipped_instances += 1
                continue

        to_sig = lambda s_str: str(sum([(2 ** i) * int(b) for (i, b) in enumerate(s_str[::-1])]))

        for side, writer_dict, seq in zip(
                [OPTS.source_lang, OPTS.target_lang],
                [writers[OPTS.source_lang], writers[OPTS.target_lang]],
                [source, target]
        ):

            if any([pr in writer_dict.keys() for pr in ['both', 'prefix']]):
                sig_str = ' '.join(
                    [f"${to_sig(chunk)}$"
                     for chunk in chunks(signatures[side][b_idx], OPTS.lsh_dim)]
                    + [OPTS.sep_token]) + ' '

                signature_counts[side][sig_str] += 1
            else:
                sig_str = ''

            if ex_strings is not None:
                ex_strings[sig_str].append(seq)

            for repr, writer in writer_dict.items():
                if repr == 'prefix':
                    seq_str = ''
                else:
                    seq_str = model_output_sentence(seq)

                if repr == 'seq':
                    out_str = seq_str
                else:
                    out_str = sig_str + seq_str
                writer.write(out_str + '\n')

    if OPTS.debug and i == 5: break

for _, dic in writers.items():
    for writ in dic.values(): writ.close()

